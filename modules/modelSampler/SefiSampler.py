import copy
import inspect
from collections.abc import Callable
import numpy as np
from tqdm import tqdm

from modules.model.SefiModel import SefiModel
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import factory
from modules.util.config.SampleConfig import SampleConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.FileType import FileType
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.ModelType import ModelType
from modules.util.enum.NoiseScheduler import NoiseScheduler
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.torch_util import torch_gc

import torch
from torch import Tensor
from torchvision import transforms


@factory.register(BaseModelSampler, ModelType.SEFI)
class SefiSampler(BaseModelSampler):
    def __init__(
            self,
            train_device: torch.device,
            temp_device: torch.device,
            model: SefiModel,
            model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)
        self.model = model
        self.model_type = model_type

    @torch.no_grad()
    def __sample_base(
            self,
            prompt: str,
            negative_prompt: str,
            height: int,
            width: int,
            seed: int,
            random_seed: bool,
            diffusion_steps: int,
            cfg_scale: float,
            noise_scheduler: NoiseScheduler,
            text_encoder_sequence_length: int | None = None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            # Retrieve models
            self.model.text_encoder_to(self.train_device)

            batch_size = 2 if cfg_scale > 1.0 else 1
            
            # Encode text
            prompt_embeds = self.model.encode_text(
                text=[prompt, negative_prompt] if batch_size == 2 else prompt,
                train_device=self.train_device,
                text_encoder_sequence_length=text_encoder_sequence_length,
            )
            text_ids = self.model.prepare_text_ids(prompt_embeds)

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # Initialize latents
            # 1. Texture noise: [1, 32, H//8, W//8] -> patchified to [1, 128, H//16, W//16]
            texture_latent = torch.randn(
                size=(1, 32, height // 8, width // 8),
                generator=generator,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            )
            texture_latent = self.model.patchify_latents(texture_latent)

            # 2. Semantic noise: [1, 16, H//16, W//16]
            semantic_latent = torch.randn(
                size=(1, 16, height // 16, width // 16),
                generator=generator,
                device=self.train_device,
                dtype=self.model.train_dtype.torch_dtype(),
            )

            # Combine
            latents = torch.cat([semantic_latent, texture_latent], dim=1)
            latent_ids = self.model.prepare_latent_image_ids(latents)

            # Schedule setup
            alpha = 0.3
            delta_t = 0.1

            u_base_unit = torch.linspace(0.0, 1.0, steps=diffusion_steps + 1, device=self.train_device)
            u_shifted_unit = (alpha * u_base_unit) / (1.0 + (alpha - 1.0) * u_base_unit)
            u_sem_raw_schedule = u_shifted_unit * (1.0 + delta_t)

            self.model.transformer_to(self.train_device)

            def get_timesteps_and_sigmas(u_val: Tensor) -> tuple[Tensor, Tensor]:
                num_steps = int(self.model.noise_scheduler.config.num_train_timesteps)
                indices = (u_val * (num_steps - 1)).long().clamp(0, num_steps - 1)
                
                timesteps = self.model.noise_scheduler.timesteps[indices.cpu()].to(self.train_device)
                sigmas = self.model.noise_scheduler.sigmas[indices.cpu()].to(
                    device=self.train_device,
                    dtype=self.model.train_dtype.torch_dtype()
                )
                while sigmas.ndim < latents.ndim:
                    sigmas = sigmas.unsqueeze(-1)
                return timesteps, sigmas

            # Denoising loop
            for step in range(diffusion_steps):
                u_sem_raw_cur = torch.tensor([u_sem_raw_schedule[step]], device=self.train_device)
                u_sem_raw_next = torch.tensor([u_sem_raw_schedule[step + 1]], device=self.train_device)

                u_tex_cur = torch.clamp(u_sem_raw_cur - delta_t, min=0.0, max=1.0)
                u_sem_cur = torch.clamp(u_sem_raw_cur, max=1.0)
                u_tex_next = torch.clamp(u_sem_raw_next - delta_t, min=0.0, max=1.0)
                u_sem_next = torch.clamp(u_sem_raw_next, max=1.0)

                timesteps_sem_cur, sigmas_sem_cur = get_timesteps_and_sigmas(u_sem_cur)
                timesteps_tex_cur, sigmas_tex_cur = get_timesteps_and_sigmas(u_tex_cur)
                _, sigmas_sem_next = get_timesteps_and_sigmas(u_sem_next)
                _, sigmas_tex_next = get_timesteps_and_sigmas(u_tex_next)

                # Prepare model input
                latent_model_input = torch.cat([latents] * batch_size)
                packed_latent_input = self.model.pack_latents(latent_model_input)

                # Expand timesteps
                timesteps_sem_expanded = timesteps_sem_cur.expand(latent_model_input.shape[0])
                timesteps_tex_expanded = timesteps_tex_cur.expand(latent_model_input.shape[0])

                # Predict velocity
                pred = self.model.transformer(
                    hidden_states=packed_latent_input.to(dtype=self.model.train_dtype.torch_dtype()),
                    timestep_sem=timesteps_sem_expanded,
                    timestep_tex=timesteps_tex_expanded,
                    encoder_hidden_states=prompt_embeds.to(dtype=self.model.train_dtype.torch_dtype()),
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                )
                pred = self.model.unpack_latents(pred, latents.shape[2], latents.shape[3])

                if batch_size == 2:
                    pred_positive, pred_negative = pred.chunk(2)
                    velocity = pred_negative + cfg_scale * (pred_positive - pred_negative)
                else:
                    velocity = pred

                # Update semantic & texture parts separately
                vel_sem = velocity[:, :16]
                vel_tex = velocity[:, 16:]
                
                lat_sem = latents[:, :16]
                lat_tex = latents[:, 16:]

                dt_sem = sigmas_sem_next - sigmas_sem_cur
                dt_tex = sigmas_tex_next - sigmas_tex_cur

                lat_sem = lat_sem + dt_sem * vel_sem
                lat_tex = lat_tex + dt_tex * vel_tex
                
                latents = torch.cat([lat_sem, lat_tex], dim=1)

                on_update_progress(step + 1, diffusion_steps)

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # Decode texture to image
            self.model.vae_to(self.train_device)

            texture_latents = latents[:, 16:]
            texture_latents = self.model.unscale_latents(texture_latents)
            raw_latents = self.model.unpatchify_latents(texture_latents)

            decoded = self.model.vae.decode(raw_latents.to(dtype=self.model.train_dtype.torch_dtype()), return_dict=False)[0]

            # Postprocess to PIL
            # Rescale to [0, 1] and map to PIL
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()
            decoded = (decoded * 255).round().astype(np.uint8)
            pil_image = transforms.ToPILImage()(decoded[0])

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=pil_image,
            )

    def sample(
            self,
            sample_config: SampleConfig,
            destination: str,
            image_format: ImageFormat | None = None,
            video_format: VideoFormat | None = None,
            audio_format: AudioFormat | None = None,
            on_sample: Callable[[ModelSamplerOutput], None] = lambda _: None,
            on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ):
        sampler_output = self.__sample_base(
            prompt=sample_config.prompt,
            negative_prompt=sample_config.negative_prompt,
            height=self.quantize_resolution(sample_config.height, 64),
            width=self.quantize_resolution(sample_config.width, 64),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            text_encoder_sequence_length=sample_config.text_encoder_1_sequence_length,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output, destination,
            image_format, video_format, audio_format,
        )

        on_sample(sampler_output)
