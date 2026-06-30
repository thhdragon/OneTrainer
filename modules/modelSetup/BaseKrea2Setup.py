from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.Krea2Model import Krea2Model
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.modelSetup.mixin.ModelSetupText2ImageMixin import ModelSetupText2ImageMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_krea2_transformer,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor

from einops import rearrange, repeat


class BaseKrea2Setup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    ModelSetupText2ImageMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "full": [],
        "blocks": ["blocks"],
        "attn-mlp": {'patterns': ["^(?=.*attn)(?!.*txtfusion).*", "^(?=.*mlp)(?!.*txtfusion).*"], 'regex': True},
        "attn-only": {'patterns': ["^(?=.*attn)(?!.*txtfusion).*"], 'regex': True},
    }

    def setup_optimizations(
            self,
            model: Krea2Model,
            config: TrainConfig,
    ):
        if config.gradient_checkpointing.enabled():
            model.transformer_offload_conductor = \
                enable_checkpointing_for_krea2_transformer(model.transformer, config)
            if model.text_encoder is not None:
                model.text_encoder_offload_conductor = \
                    enable_checkpointing_for_qwen3_encoder_layers(model.text_encoder, config)

        model.autocast_context, model.train_dtype = create_autocast_context(self.train_device, config.train_dtype, [
            config.weight_dtypes().transformer,
            config.weight_dtypes().text_encoder,
            config.weight_dtypes().vae,
            config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
        ], config.enable_autocast_cache)

        model.text_encoder_autocast_context, model.text_encoder_train_dtype = \
            disable_fp16_autocast_context(
                self.train_device,
                config.train_dtype,
                config.fallback_train_dtype,
                [
                    config.weight_dtypes().text_encoder,
                    config.weight_dtypes().lora if config.training_method == TrainingMethod.LORA else None,
                ],
                config.enable_autocast_cache,
            )

        quantize_layers(model.text_encoder, self.train_device, model.text_encoder_train_dtype, config)
        quantize_layers(model.vae, self.train_device, model.train_dtype, config)
        quantize_layers(model.transformer, self.train_device, model.train_dtype, config)

    def predict(
            self,
            model: Krea2Model,
            batch: dict,
            config: TrainConfig,
            train_progress: TrainProgress,
            *,
            deterministic: bool = False,
    ) -> dict:
        with model.autocast_context:
            batch_seed = 0 if deterministic else train_progress.global_step * multi.world_size() + multi.rank()
            generator = torch.Generator(device=config.train_device)
            generator.manual_seed(batch_seed)
            rand = Random(batch_seed)

            # Get text hidden states and mask
            if batch.get('text_encoder_hidden_state') is not None:
                context = batch.get('text_encoder_hidden_state')
                txtmask = batch.get('tokens_mask').bool()
            else:
                context, txtmask = model.encode_text(
                    train_device=self.train_device,
                    batch_size=batch['latent_image'].shape[0],
                    rand=rand,
                    tokens=batch.get("tokens"),
                    tokens_mask=batch.get("tokens_mask"),
                    text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
                )

            max_txt_len = max(1, txtmask.sum(dim=1).max().item())
            context = context[:, :max_txt_len]
            txtmask = txtmask[:, :max_txt_len]

            scaled_latent_image = model.scale_latents(batch['latent_image'])
            latent_noise = self._create_noise(scaled_latent_image, config, generator)

            shift = model.calculate_timestep_shift(scaled_latent_image.shape[-2], scaled_latent_image.shape[-1])
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            scaled_noisy_latent_image, sigma = self._add_noise_discrete(
                scaled_latent_image,
                latent_noise,
                timestep,
                model.noise_scheduler.timesteps,
            )

            # Patchify image latent: [B, C, H, W] -> [B, h*w, C*ph*pw]
            patch = model.transformer.patch
            h_, w_ = scaled_noisy_latent_image.shape[-2] // patch, scaled_noisy_latent_image.shape[-1] // patch
            img_tokens = rearrange(scaled_noisy_latent_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

            bsize, imglen, _ = img_tokens.shape

            # imgids and imgmask
            imgids = torch.zeros((h_, w_, 3), device=self.train_device)
            imgids[..., 1] = torch.arange(h_, device=self.train_device)[:, None]
            imgids[..., 2] = torch.arange(w_, device=self.train_device)[None, :]
            imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
            imgmask = torch.ones(bsize, imglen, device=self.train_device, dtype=torch.bool)

            # txtpos
            txtpos = torch.zeros(bsize, max_txt_len, 3, device=self.train_device)

            # combine
            mask = torch.cat((imgmask, txtmask), dim=1)
            pos = torch.cat((imgpos, txtpos), dim=1)

            output = model.transformer(
                img=img_tokens.to(dtype=model.train_dtype.torch_dtype()),
                context=context.to(dtype=model.train_dtype.torch_dtype()),
                t=(timestep / 1000.0).to(device=self.train_device),
                pos=pos.to(device=self.train_device),
                mask=mask.to(device=self.train_device),
                return_dict=True
            ).sample

            # Unpatchify predicted flow back to [B, C, H, W]
            predicted_flow = rearrange(output, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)

            flow = latent_noise - scaled_latent_image
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': predicted_flow,
                'target': flow,
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = scaled_noisy_latent_image - predicted_flow * sigma
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", latent_noise, config, train_progress)
                    self._save_latent("2-noisy_image", scaled_noisy_latent_image, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow, config, train_progress)
                    self._save_latent("4-flow", flow, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: Krea2Model,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        return self._flow_matching_losses(
            batch=batch,
            data=data,
            config=config,
            train_device=self.train_device,
            sigmas=model.noise_scheduler.sigmas,
        ).mean()

    def prepare_text_caching(self, model: Krea2Model, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)

        model.eval()
        torch_gc()
