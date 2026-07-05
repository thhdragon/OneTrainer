import os

from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.mixin.DataLoaderText2ImageMixin import DataLoaderText2ImageMixin
from modules.model.Flux2Model import (
    QWEN3_HIDDEN_STATES_LAYERS,
    qwen3_format_input,
)
from modules.model.SefiModel import SefiModel
from modules.modelSetup.SefiSetup import SefiSetup
from modules.dataLoader.sefi.EncodeSefiSemantic import EncodeSefiSemantic
from modules.util import factory
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.TrainProgress import TrainProgress

from mgds.pipelineModules.DecodeTokens import DecodeTokens
from mgds.pipelineModules.DecodeVAE import DecodeVAE
from mgds.pipelineModules.EncodeQwenText import EncodeQwenText
from mgds.pipelineModules.EncodeVAE import EncodeVAE
from mgds.pipelineModules.RescaleImageChannels import RescaleImageChannels
from mgds.pipelineModules.SampleVAEDistribution import SampleVAEDistribution
from mgds.pipelineModules.SaveImage import SaveImage
from mgds.pipelineModules.SaveText import SaveText
from mgds.pipelineModules.ScaleImage import ScaleImage
from mgds.pipelineModules.Tokenize import Tokenize


@factory.register(BaseDataLoader, ModelType.SEFI)
class SefiDataLoader(
    BaseDataLoader,
    DataLoaderText2ImageMixin,
):
    def _preparation_modules(self, config: TrainConfig, model: SefiModel):
        # 1. Texture VAE Encoding
        rescale_image = RescaleImageChannels(image_in_name='image', image_out_name='image', in_range_min=0, in_range_max=1, out_range_min=-1, out_range_max=1)
        encode_image = EncodeVAE(in_name='image', out_name='latent_image_distribution', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        image_sample = SampleVAEDistribution(in_name='latent_image_distribution', out_name='latent_image', mode='mean')
        
        # 2. Semantic SemVAE Encoding
        encode_semantic = EncodeSefiSemantic(
            in_name='image',  # Reads the raw image in [0, 1] range before rescale
            out_name='semantic_latent',
            dino_encoder=model.dino_encoder,
            semvae=model.semvae,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype(),
        )

        downscale_mask = ScaleImage(in_name='mask', out_name='latent_mask', factor=0.125)

        # 3. Text Tokenization and Encoding
        tokenize_prompt = Tokenize(
            in_name='prompt',
            tokens_out_name='tokens',
            mask_out_name='tokens_mask',
            tokenizer=model.tokenizer,
            max_token_length=config.text_encoder_sequence_length,
            apply_chat_template=lambda caption: qwen3_format_input(caption),
            apply_chat_template_kwargs={'add_generation_prompt': True, 'enable_thinking': False}
        )
        encode_prompt = EncodeQwenText(
            tokens_name='tokens',
            tokens_attention_mask_in_name='tokens_mask',
            hidden_state_out_name='text_encoder_hidden_state',
            tokens_attention_mask_out_name='tokens_mask',
            text_encoder=model.text_encoder,
            hidden_state_output_index=QWEN3_HIDDEN_STATES_LAYERS,
            autocast_contexts=[model.autocast_context],
            dtype=model.train_dtype.torch_dtype()
        )

        modules = [rescale_image, encode_image, image_sample, encode_semantic]
        if config.masked_training or config.model_type.has_mask_input():
            modules.append(downscale_mask)

        modules += [tokenize_prompt, encode_prompt]
        return modules

    def _cache_modules(self, config: TrainConfig, model: SefiModel, model_setup: SefiSetup):
        image_split_names = ['latent_image', 'semantic_latent', 'original_resolution', 'crop_offset']

        if config.masked_training or config.model_type.has_mask_input():
            image_split_names.append('latent_mask')

        image_aggregate_names = ['crop_resolution', 'image_path']
        text_split_names = []

        sort_names = image_aggregate_names + image_split_names + [
            'prompt', 'tokens', 'tokens_mask', 'text_encoder_hidden_state',
            'concept'
        ]

        text_split_names += ['tokens', 'tokens_mask', 'text_encoder_hidden_state']

        return self._cache_modules_from_names(
            model, model_setup,
            image_split_names=image_split_names,
            image_aggregate_names=image_aggregate_names,
            text_split_names=text_split_names,
            sort_names=sort_names,
            config=config,
            text_caching=True,
        )

    def _output_modules(self, config: TrainConfig, model: SefiModel, model_setup: SefiSetup):
        output_names = [
            'image_path', 'latent_image', 'semantic_latent',
            'prompt',
            'tokens',
            'tokens_mask',
            'original_resolution', 'crop_resolution', 'crop_offset',
        ]

        if config.masked_training or config.model_type.has_mask_input():
            output_names.append('latent_mask')

        output_names.append('text_encoder_hidden_state')

        return self._output_modules_from_out_names(
            model, model_setup,
            output_names=output_names,
            config=config,
            use_conditioning_image=False,
            vae=model.vae,
            autocast_context=[model.autocast_context],
            train_dtype=model.train_dtype,
        )

    def _debug_modules(self, config: TrainConfig, model: SefiModel):
        debug_dir = os.path.join(config.debug_dir, "dataloader")

        def before_save_fun():
            model.vae_to(self.train_device)

        decode_image = DecodeVAE(in_name='latent_image', out_name='decoded_image', vae=model.vae, autocast_contexts=[model.autocast_context], dtype=model.train_dtype.torch_dtype())
        upscale_mask = ScaleImage(in_name='latent_mask', out_name='decoded_mask', factor=8)
        decode_prompt = DecodeTokens(in_name='tokens', out_name='decoded_prompt', tokenizer=model.tokenizer)
        save_image = SaveImage(image_in_name='decoded_image', original_path_in_name='image_path', path=debug_dir, in_range_min=-1, in_range_max=1, before_save_fun=before_save_fun)
        save_mask = SaveImage(image_in_name='decoded_mask', original_path_in_name='image_path', path=debug_dir, in_range_min=0, in_range_max=1, before_save_fun=before_save_fun)
        save_prompt = SaveText(text_in_name='decoded_prompt', original_path_in_name='image_path', path=debug_dir, before_save_fun=before_save_fun)

        modules = []
        modules.append(decode_image)
        modules.append(save_image)

        if config.masked_training or config.model_type.has_mask_input():
            modules.append(upscale_mask)
            modules.append(save_mask)

        modules.append(decode_prompt)
        modules.append(save_prompt)

        return modules

    def _create_dataset(
            self,
            config: TrainConfig,
            model: SefiModel,
            model_setup: SefiSetup,
            train_progress: TrainProgress,
            is_validation: bool = False,
    ):
        return DataLoaderText2ImageMixin._create_dataset(
            self,
            config, model, model_setup, train_progress, is_validation,
            aspect_bucketing_quantization=64,
        )
