import os
import sys
import torch
import traceback
from omegaconf import OmegaConf

# Append SeFi-Image to path to import config & builder
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image", "SFD"))

from sefi.config import load_config
from sefi.builder import build_components
from sefi.runner import _resolve_checkpoint_file, _load_checkpoint_payload, _extract_state_dict, _strip_prefix_if_needed
from tokenizer.semvae.models.vae import SemanticVariationalAutoEncoder

from modules.model.SefiModel import SefiModel
from modules.modelLoader.GenericFineTuneModelLoader import make_fine_tune_model_loader
from modules.modelLoader.GenericLoRAModelLoader import make_lora_model_loader
from modules.modelLoader.mixin.HFModelLoaderMixin import HFModelLoaderMixin
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.util.config.TrainConfig import QuantizationConfig
from modules.util.enum.ModelType import ModelType
from modules.util.ModelNames import ModelNames
from modules.util.ModelWeightDtypes import ModelWeightDtypes


class SefiModelLoader(HFModelLoaderMixin):
    def __init__(self):
        super().__init__()

    def load(
            self,
            model: SefiModel,
            model_type: ModelType,
            model_names: ModelNames,
            weight_dtypes: ModelWeightDtypes,
            quantization: QuantizationConfig,
    ):
        base_model_name = model_names.base_model
        config_path = os.path.join(base_model_name, "sefi_config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image", "sefi_config.yaml")

        print(f"Loading SEFI model from config: {config_path}")
        resolved_config = load_config(config_path)

        # Force weights root to be the base model directory if they are relative
        model_cfg = resolved_config.model
        if model_cfg.assets.transformer_config_path == ".":
            model_cfg.assets.transformer_config_path = base_model_name
        if model_cfg.assets.scheduler_path == ".":
            model_cfg.assets.scheduler_path = base_model_name
        if model_cfg.texture_vae.base_path == ".":
            model_cfg.texture_vae.base_path = base_model_name
        if model_cfg.text_encoder.weights_root == ".":
            model_cfg.text_encoder.weights_root = base_model_name

        # Build components using sefi package builder
        dtype = torch.bfloat16 if weight_dtypes.train_dtype.torch_dtype() is None else weight_dtypes.train_dtype.torch_dtype()
        components = build_components(resolved_config, component_dtype=dtype)

        # Load transformer weights separately
        transformer_ckpt_dir = base_model_name
        ckpt_file = _resolve_checkpoint_file(transformer_ckpt_dir)
        print(f"Loading transformer checkpoint: {ckpt_file}")
        payload = _load_checkpoint_payload(ckpt_file)
        state_dict = _extract_state_dict(payload)
        state_dict = _strip_prefix_if_needed(state_dict, "module.")

        missing, unexpected = components.transformer.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: transformer missing keys: {missing[:10]}")
        if unexpected:
            print(f"Warning: transformer unexpected keys: {unexpected[:10]}")

        # Set up SemVAE
        semvae_ckpt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SeFi-Image", "semVAE", "dinov2_vitl14_reg", "transformer_ch16", "checkpoints", "checkpoint_01000000.pt")
        if not os.path.exists(semvae_ckpt_path):
            raise FileNotFoundError(f"SemVAE checkpoint not found at: {semvae_ckpt_path}")

        print(f"Loading SemVAE model from checkpoint: {semvae_ckpt_path}")
        semvae = SemanticVariationalAutoEncoder(
            input_dim=1024,
            bottleneck_dim=16,
            hidden_dim=1024,
            arch='transformer',
            transformer_heads=16,
            transformer_blocks=4
        )
        semvae_ckpt = torch.load(semvae_ckpt_path, map_location='cpu')
        semvae_state_dict = semvae_ckpt['model_state_dict'] if 'model_state_dict' in semvae_ckpt else semvae_ckpt
        semvae_state_dict = {k.replace('module.', ''): v for k, v in semvae_state_dict.items()}
        semvae.load_state_dict(semvae_state_dict)
        semvae.eval()

        # Set up DINOv2 encoder
        print("Loading DINOv2-L vision backbone...")
        dino_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
        dino_encoder.eval()

        model.model_type = model_type
        model.tokenizer = components.text_encoder.tokenizer
        model.noise_scheduler = components.noise_scheduler
        model.text_encoder = components.text_encoder
        model.vae = components.texture_codec.texture_vae
        model.semvae = semvae
        model.dino_encoder = dino_encoder
        model.transformer = components.transformer


class SefiLoRALoader(LoRALoaderMixin):
    def __init__(self):
        super().__init__()

    def load(self, model: SefiModel, model_names: ModelNames):
        return self._load(model, model_names)


SefiLoRAModelLoader = make_lora_model_loader(
    model_spec_map={
        ModelType.SEFI: "resources/sd_model_spec/flux_2.0-lora.json", # Reuse flux model spec mappings
    },
    model_class=SefiModel,
    model_loader_class=SefiModelLoader,
    lora_loader_class=SefiLoRALoader,
    embedding_loader_class=None,
)

SefiFineTuneModelLoader = make_fine_tune_model_loader(
    model_spec_map={
        ModelType.SEFI: "resources/sd_model_spec/flux_2.0.json",
    },
    model_class=SefiModel,
    model_loader_class=SefiModelLoader,
    embedding_loader_class=None,
)
