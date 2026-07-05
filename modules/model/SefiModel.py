import math
import sys
import os
from contextlib import nullcontext
from random import Random

# Append SeFi-Image to path to import model components
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image", "SFD"))

from modules.model.BaseModel import BaseModel
from modules.module.LoRAModule import LoRAModuleWrapper
from modules.util.convert_util import chunk_swap
from modules.util.enum.ModelType import ModelType
from modules.util.LayerOffloadConductor import LayerOffloadConductor

import torch
from torch import Tensor

from diffusers import (
    AutoencoderKLFlux2,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from sefi.modeling.flux2_sefi_transformer import Flux2SEFITransformer2DModel
from sefi.modeling.qwen3vl_text_encoder import Qwen3VLTextEncoder
from tokenizer.semvae.models.vae import SemanticVariationalAutoEncoder


class SefiModel(BaseModel):
    tokenizer: object | None
    noise_scheduler: FlowMatchEulerDiscreteScheduler | None
    text_encoder: Qwen3VLTextEncoder | None
    vae: AutoencoderKLFlux2 | None
    semvae: SemanticVariationalAutoEncoder | None
    dino_encoder: torch.nn.Module | None
    transformer: Flux2SEFITransformer2DModel | None

    text_encoder_autocast_context: torch.autocast | nullcontext

    text_encoder_offload_conductor: LayerOffloadConductor | None
    transformer_offload_conductor: LayerOffloadConductor | None

    transformer_lora: LoRAModuleWrapper | None
    lora_state_dict: dict | None

    def __init__(self, model_type: ModelType):
        super().__init__(model_type=model_type)

        self.tokenizer = None
        self.noise_scheduler = None
        self.text_encoder = None
        self.vae = None
        self.semvae = None
        self.dino_encoder = None
        self.transformer = None

        self.text_encoder_autocast_context = nullcontext()

        self.text_encoder_offload_conductor = None
        self.transformer_offload_conductor = None

        self.transformer_lora = None
        self.lora_state_dict = None

    def adapters(self) -> list[LoRAModuleWrapper]:
        return [a for a in [self.transformer_lora] if a is not None]

    def fusion_groups(self) -> list | None:
        # Map parameters of wrapped backbone.
        return [
            ("transformer.backbone.transformer_blocks.{i}", ["attn.to_q", "attn.to_k", "attn.to_v"], "attn.qkv", "img_attn.qkv"),
            ("transformer.backbone.transformer_blocks.{i}", ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"], "attn.added_qkv", "txt_attn.qkv"),
        ]

    def diffusers_to_original(self) -> list | None:
        # Prefix keys with backbone. because Flux2SEFITransformer2DModel wraps Flux2Transformer2DModel
        return [
            ("backbone.context_embedder", "txt_in"),
            ("backbone.x_embedder",       "img_in"),
            ("backbone.time_guidance_embed.timestep_embedder", "time_in", [
                ("linear_1", "in_layer"),
                ("linear_2", "out_layer"),
            ]),
            ("backbone.time_guidance_embed.guidance_embedder", "guidance_in", [
                ("linear_1", "in_layer"),
                ("linear_2", "out_layer"),
            ]),
            ("backbone.double_stream_modulation_img.linear", "double_stream_modulation_img.lin"),
            ("backbone.double_stream_modulation_txt.linear", "double_stream_modulation_txt.lin"),
            ("backbone.single_stream_modulation.linear",     "single_stream_modulation.lin"),
            ("backbone.proj_out",                            "final_layer.linear"),
            *chunk_swap("backbone.norm_out.linear", "final_layer.adaLN_modulation.1"),
            ("backbone.transformer_blocks.{i}", "double_blocks.{i}", [
                ("attn.qkv",                 "img_attn.qkv"),
                ("attn.added_qkv",           "txt_attn.qkv"),
                ("attn.norm_k.weight",       "img_attn.norm.key_norm.scale"),
                ("attn.norm_q.weight",       "img_attn.norm.query_norm.scale"),
                ("attn.to_out.0",            "img_attn.proj"),
                ("ff.linear_in",             "img_mlp.0"),
                ("ff.linear_out",            "img_mlp.2"),
                ("attn.norm_added_k.weight", "txt_attn.norm.key_norm.scale"),
                ("attn.norm_added_q.weight", "txt_attn.norm.query_norm.scale"),
                ("attn.to_add_out",          "txt_attn.proj"),
                ("ff_context.linear_in",     "txt_mlp.0"),
                ("ff_context.linear_out",    "txt_mlp.2"),
            ]),
            ("backbone.single_transformer_blocks.{i}", "single_blocks.{i}", [
                ("attn.to_qkv_mlp_proj", "linear1"),
                ("attn.to_out",          "linear2"),
                ("attn.norm_k.weight",   "norm.key_norm.scale"),
                ("attn.norm_q.weight",   "norm.query_norm.scale"),
            ]),
        ]

    def vae_to(self, device: torch.device):
        self.vae.to(device=device)

    def text_encoder_to(self, device: torch.device):
        if self.text_encoder is not None:
            if self.text_encoder_offload_conductor is not None:
                self.text_encoder_offload_conductor.to(device)
            else:
                self.text_encoder.to(device=device)

    def transformer_to(self, device: torch.device):
        if self.transformer_offload_conductor is not None:
            self.transformer_offload_conductor.to(device)
        else:
            self.transformer.to(device=device)

        if self.transformer_lora is not None:
            self.transformer_lora.to(device)

    def to(self, device: torch.device):
        self.vae_to(device)
        self.text_encoder_to(device)
        self.transformer_to(device)
        if self.semvae is not None:
            self.semvae.to(device=device)
        if self.dino_encoder is not None:
            self.dino_encoder.to(device=device)

    def eval(self):
        self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        self.transformer.eval()
        if self.semvae is not None:
            self.semvae.eval()
        if self.dino_encoder is not None:
            self.dino_encoder.eval()

    def create_pipeline(self) -> DiffusionPipeline:
        # Standard SEFI pipeline wrapper
        from sefi import SEFIInferencePipeline
        # We can implement or wrap the pipeline creation if needed, or return None if training only.
        return None

    def encode_text(
            self,
            train_device: torch.device,
            batch_size: int = 1,
            rand: Random | None = None,
            text: str = None,
            tokens: Tensor = None,
            tokens_mask: Tensor = None,
            text_encoder_sequence_length: int | None = None,
            text_encoder_dropout_probability: float | None = None,
            text_encoder_output: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        if tokens is None and text is not None:
            if isinstance(text, str):
                text = [text]

            # Qwen3VLTextEncoder encodes text to embeds on the fly
            with self.text_encoder_autocast_context:
                embeds, _ = self.text_encoder.encode(text, dtype=self.text_encoder.model.dtype)
            return embeds

        if text_encoder_output is not None:
            return text_encoder_output

        # If tokens are provided (from cache), run text encoder:
        with self.text_encoder_autocast_context:
            output = self.text_encoder.model.model(
                input_ids=tokens.to(self.text_encoder.model.device),
                attention_mask=tokens_mask.to(self.text_encoder.model.device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = output.hidden_states
            stacked = torch.stack([hidden_states[idx] for idx in self.text_encoder.hidden_layers], dim=1)
            stacked = stacked.to(dtype=self.text_encoder.model.dtype)
            batch_size, num_layers, seq_len, hidden_dim = stacked.shape
            prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_layers * hidden_dim)
        return prompt_embeds

    @staticmethod
    def prepare_latent_image_ids(latents: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = latents.shape
        t = torch.arange(1, device=latents.device)
        h = torch.arange(height, device=latents.device)
        w = torch.arange(width, device=latents.device)
        l_ = torch.arange(1, device=latents.device)
        latent_ids = torch.cartesian_prod(t, h, w, l_)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    @staticmethod
    def pack_latents(latents) -> Tensor:
        batch_size, num_channels, height, width = latents.shape
        return latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    @staticmethod
    def unpack_latents(latents, height: int, width: int) -> Tensor:
        batch_size, seq_len, num_channels = latents.shape
        return latents.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)

    def calculate_timestep_shift(self, latent_height: int, latent_width: int) -> float:
        # Standard shift
        base_seq_len = self.noise_scheduler.config.base_image_seq_len
        max_seq_len = self.noise_scheduler.config.max_image_seq_len
        base_shift = self.noise_scheduler.config.base_shift
        max_shift = self.noise_scheduler.config.max_shift
        patch_size = 2

        image_seq_len = (latent_width // patch_size) * (latent_height // patch_size)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return math.exp(mu)

    @staticmethod
    def prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        out_ids = []
        for _ in range(B):
            t = torch.arange(1, device=x.device)
            h = torch.arange(1, device=x.device)
            w = torch.arange(1, device=x.device)
            l_ = torch.arange(L, device=x.device)
            coords = torch.cartesian_prod(t, h, w, l_)
            out_ids.append(coords)
        return torch.stack(out_ids)

    @staticmethod
    def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    def scale_latents(self, latents: Tensor) -> Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return (latents - latents_bn_mean) / latents_bn_std

    def unscale_latents(self, latents: Tensor) -> Tensor:
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return latents * latents_bn_std + latents_bn_mean
