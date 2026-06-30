import copy
import inspect
import math
from collections.abc import Callable

from modules.model.Krea2Model import Krea2Model
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

from diffusers.image_processor import VaeImageProcessor

from einops import rearrange, repeat
from tqdm import tqdm


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0):
    ts = torch.linspace(1, 0, steps + 1)
    slope = (y2 - y1) / (x2 - x1)
    mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@factory.register(BaseModelSampler, ModelType.KREA2)
class Krea2Sampler(BaseModelSampler):
    def __init__(
        self,
        train_device: torch.device,
        temp_device: torch.device,
        model: Krea2Model,
        model_type: ModelType,
    ):
        super().__init__(train_device, temp_device)

        self.model = model
        self.model_type = model_type
        self.image_processor = VaeImageProcessor(vae_scale_factor=8)

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
        on_update_progress: Callable[[int, int], None] = lambda _, __: None,
    ) -> ModelSamplerOutput:
        with self.model.autocast_context:
            generator = torch.Generator(device=self.train_device)
            if random_seed:
                generator.seed()
            else:
                generator.manual_seed(seed)

            transformer = self.model.transformer
            vae = self.model.vae

            # prepare prompt
            self.model.text_encoder_to(self.train_device)

            do_cfg = cfg_scale > 1.0 and negative_prompt is not None

            txt, txtmask = self.model.encode_text(
                text=prompt,
                train_device=self.train_device,
            )

            if do_cfg:
                untxt, untxtmask = self.model.encode_text(
                    text=negative_prompt,
                    train_device=self.train_device,
                )

            self.model.text_encoder_to(self.temp_device)
            torch_gc()

            # prepare latent noise
            latent_image = torch.randn(
                size=(1, 16, height // 8, width // 8),
                generator=generator,
                device=self.train_device,
                dtype=torch.float32,
            )

            # prepare img tokens
            patch = transformer.patch
            h_, w_ = (height // 8) // patch, (width // 8) // patch
            img_tokens = rearrange(latent_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

            bsize, imglen, _ = img_tokens.shape

            # imgids and imgmask
            imgids = torch.zeros((h_, w_, 3), device=self.train_device)
            imgids[..., 1] = torch.arange(h_, device=self.train_device)[:, None]
            imgids[..., 2] = torch.arange(w_, device=self.train_device)[None, :]
            imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsize, three=3)
            imgmask = torch.ones(bsize, imglen, device=self.train_device, dtype=torch.bool)

            # txtpos and mask
            txtpos = torch.zeros(bsize, txt.shape[1], 3, device=self.train_device)
            pos = torch.cat((imgpos, txtpos), dim=1)
            mask = torch.cat((imgmask, txtmask), dim=1)

            if do_cfg:
                untxtpos = torch.zeros(bsize, untxt.shape[1], 3, device=self.train_device)
                unpos = torch.cat((imgpos, untxtpos), dim=1)
                unmask = torch.cat((imgmask, untxtmask), dim=1)

            # timesteps
            align = 16
            x1 = (256 // align) ** 2
            x2 = (1280 // align) ** 2
            ts = timesteps(imglen, diffusion_steps, x1, x2)

            self.model.transformer_to(self.train_device)

            img = img_tokens.to(device=self.train_device, dtype=self.model.train_dtype.torch_dtype())
            txt = txt.to(device=self.train_device, dtype=self.model.train_dtype.torch_dtype())
            pos = pos.to(device=self.train_device)
            mask = mask.to(device=self.train_device)
            if do_cfg:
                untxt = untxt.to(device=self.train_device, dtype=self.model.train_dtype.torch_dtype())
                unpos = unpos.to(device=self.train_device)
                unmask = unmask.to(device=self.train_device)

            for i, (tcurr, tprev) in enumerate(tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="sampling")):
                t = torch.full((1,), tcurr, dtype=img.dtype, device=self.train_device)
                cond = transformer(img=img, context=txt, t=t, pos=pos, mask=mask, return_dict=True).sample
                if do_cfg:
                    uncond = transformer(img=img, context=untxt, t=t, pos=unpos, mask=unmask, return_dict=True).sample
                    v = uncond + cfg_scale * (cond - uncond)
                else:
                    v = cond
                img = img + (tprev - tcurr) * v

                on_update_progress(i + 1, len(ts) - 1)

            self.model.transformer_to(self.temp_device)
            torch_gc()

            # Unpatchify back to latent [1, 16, height//8, width//8]
            latent = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)
            latent = latent.unsqueeze(2)  # add frame dim: [1, 16, 1, height//8, width//8]

            self.model.vae_to(self.train_device)

            latents = self.model.unscale_latents(latent)
            image = vae.decode(latents, return_dict=False)[0].squeeze(-3)

            do_denormalize = [True] * image.shape[0]
            image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=do_denormalize)

            self.model.vae_to(self.temp_device)
            torch_gc()

            return ModelSamplerOutput(
                file_type=FileType.IMAGE,
                data=image[0],
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
            height=self.quantize_resolution(sample_config.height, 16),
            width=self.quantize_resolution(sample_config.width, 16),
            seed=sample_config.seed,
            random_seed=sample_config.random_seed,
            diffusion_steps=sample_config.diffusion_steps,
            cfg_scale=sample_config.cfg_scale,
            noise_scheduler=sample_config.noise_scheduler,
            on_update_progress=on_update_progress,
        )

        self.save_sampler_output(
            sampler_output,
            destination,
            image_format,
            video_format,
            audio_format,
        )

        on_sample(sampler_output)
