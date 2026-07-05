from abc import ABCMeta
from random import Random

import modules.util.multi_gpu_util as multi
from modules.model.SefiModel import SefiModel
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.modelSetup.mixin.ModelSetupDebugMixin import ModelSetupDebugMixin
from modules.modelSetup.mixin.ModelSetupDiffusionLossMixin import ModelSetupDiffusionLossMixin
from modules.modelSetup.mixin.ModelSetupEmbeddingMixin import ModelSetupEmbeddingMixin
from modules.modelSetup.mixin.ModelSetupFlowMatchingMixin import ModelSetupFlowMatchingMixin
from modules.modelSetup.mixin.ModelSetupNoiseMixin import ModelSetupNoiseMixin
from modules.util.checkpointing_util import (
    enable_checkpointing_for_flux2_transformer,
    enable_checkpointing_for_qwen3_encoder_layers,
)
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_autocast_context, disable_fp16_autocast_context
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.quantization_util import quantize_layers
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress
from modules.util import factory
from modules.util.enum.ModelType import ModelType

import torch
from torch import Tensor


@factory.register(BaseModelSetup, ModelType.SEFI)
class SefiSetup(
    BaseModelSetup,
    ModelSetupDiffusionLossMixin,
    ModelSetupDebugMixin,
    ModelSetupNoiseMixin,
    ModelSetupFlowMatchingMixin,
    ModelSetupEmbeddingMixin,
    metaclass=ABCMeta
):
    LAYER_PRESETS = {
        "blocks": ["transformer_block"],
        "full": [],
    }

    def setup_optimizations(
            self,
            model: SefiModel,
            config: TrainConfig,
    ):
        model.transformer_offload_conductor = enable_checkpointing_for_flux2_transformer(model.transformer, config, config.transformer)
        model.text_encoder_offload_conductor = enable_checkpointing_for_qwen3_encoder_layers(model.text_encoder, config, config.text_encoder)

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

        self._set_attention_backend(model.transformer, config.attention_mechanism, mask=False)

    def predict(
            self,
            model: SefiModel,
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

            # 1. Encode text
            text_encoder_output = model.encode_text(
                train_device=self.train_device,
                batch_size=batch['latent_image'].shape[0],
                rand=rand,
                tokens=batch.get("tokens"),
                tokens_mask=batch.get("tokens_mask"),
                text_encoder_sequence_length=config.text_encoder_sequence_length,
                text_encoder_output=batch.get('text_encoder_hidden_state'),
                text_encoder_dropout_probability=config.text_encoder.dropout_probability if not deterministic else None,
            )

            # 2. Get texture latents & scale them
            latent_image = model.patchify_latents(batch['latent_image'].float())
            scaled_latent_image = model.scale_latents(latent_image)
            
            # 3. Get semantic latents
            semantic_latent = batch['semantic_latent'].float().to(device=scaled_latent_image.device)

            # 4. Timestep scheduling
            latent_height = latent_image.shape[-2]
            latent_width = latent_image.shape[-1]
            shift = model.calculate_timestep_shift(latent_height, latent_width)
            
            timestep = self._get_timestep_discrete(
                model.noise_scheduler.config['num_train_timesteps'],
                deterministic,
                generator,
                scaled_latent_image.shape[0],
                config,
                shift = shift if config.dynamic_timestep_shifting else config.timestep_shift,
            )

            # Convert discrete timestep to [0, 1] range
            t = timestep.float() / 1000.0

            # Dual timestep math
            delta_t = 0.1 # default
            t_sem = t * (1.0 + delta_t)
            t_tex = t_sem - delta_t
            t_sem = torch.clamp(t_sem, max=1.0)
            t_tex = torch.clamp(t_tex, min=0.0)

            # 5. Add noise independently
            noise_sem = self._create_noise(semantic_latent, config, generator)
            noise_tex = self._create_noise(scaled_latent_image, config, generator)

            t_sem_4d = t_sem.view(-1, 1, 1, 1)
            t_tex_4d = t_tex.view(-1, 1, 1, 1)

            xt_sem = (1.0 - t_sem_4d) * noise_sem + t_sem_4d * semantic_latent
            xt_tex = (1.0 - t_tex_4d) * noise_tex + t_tex_4d * scaled_latent_image

            # Combine semantic & texture along channel dimension
            xt = torch.cat([xt_sem, xt_tex], dim=1)

            # Target flow: noise - clean
            flow_sem = noise_sem - semantic_latent
            flow_tex = noise_tex - scaled_latent_image
            flow = torch.cat([flow_sem, flow_tex], dim=1)

            # 6. Setup DiT inputs
            if model.transformer.config.guidance_embeds:
                guidance = torch.tensor([config.transformer.guidance_scale], device=self.train_device, dtype=model.train_dtype.torch_dtype())
                guidance = guidance.expand(xt.shape[0])
            else:
                guidance = None

            text_ids = model.prepare_text_ids(text_encoder_output)
            image_ids = model.prepare_latent_image_ids(xt)
            packed_latent_input = model.pack_latents(xt)

            # 7. Model call
            packed_predicted_flow = model.transformer(
                hidden_states=packed_latent_input.to(dtype=model.train_dtype.torch_dtype()),
                timestep_sem=t_sem,
                timestep_tex=t_tex,
                encoder_hidden_states=text_encoder_output.to(dtype=model.train_dtype.torch_dtype()),
                txt_ids=text_ids,
                img_ids=image_ids,
                joint_attention_kwargs=None,
            )

            predicted_flow = model.unpack_latents(
                packed_predicted_flow,
                xt.shape[2],
                xt.shape[3],
            )

            # Unpatchify for loss calculation and visualization
            model_output_data = {
                'loss_type': 'target',
                'timestep': timestep,
                'predicted': model.unpatchify_latents(predicted_flow),
                'target': model.unpatchify_latents(flow),
            }

            if config.debug_mode:
                with torch.no_grad():
                    predicted_scaled_latent_image = xt_tex - predicted_flow[:, 16:] * (1.0 - t_tex_4d)
                    self._save_tokens("7-prompt", batch['tokens'], model.tokenizer, config, train_progress)
                    self._save_latent("1-noise", noise_tex, config, train_progress)
                    self._save_latent("2-noisy_image", xt_tex, config, train_progress)
                    self._save_latent("3-predicted_flow", predicted_flow[:, 16:], config, train_progress)
                    self._save_latent("4-flow", flow_tex, config, train_progress)
                    self._save_latent("5-predicted_image", predicted_scaled_latent_image, config, train_progress)
                    self._save_latent("6-image", scaled_latent_image, config, train_progress)

        return model_output_data

    def calculate_loss(
            self,
            model: SefiModel,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ) -> Tensor:
        predicted = data['predicted']
        target = data['target']

        loss_val = (predicted - target) ** 2

        # Split into semantic and texture channels:
        # Patchify is (B, 36, H, W) -> (B, 144, H//2, W//2)
        # So the first 4 channels are semantic (16 // 4 = 4), the remaining 32 are texture
        sem_loss = loss_val[:, :4]
        tex_loss = loss_val[:, 4:]

        # Weight semantic loss differently if needed, default is 1.0
        weighted_loss = torch.cat([sem_loss, tex_loss], dim=1)

        if config.masked_training and 'latent_mask' in batch:
            mask = batch['latent_mask']
            weighted_loss = weighted_loss * mask

        return weighted_loss.mean()

    def prepare_text_caching(self, model: SefiModel, config: TrainConfig):
        model.to(self.temp_device)
        model.text_encoder_to(self.train_device)
        model.eval()
        torch_gc()
