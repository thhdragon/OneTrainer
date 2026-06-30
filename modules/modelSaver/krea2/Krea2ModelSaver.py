import os
import shutil
from pathlib import Path

from modules.model.Krea2Model import Krea2Model
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.enum.ModelFormat import ModelFormat

import torch

from safetensors.torch import save_file


class Krea2ModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_diffusers(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        # Save the transformer sub-module using its save_pretrained
        transformer_dest = os.path.join(destination, "transformer")
        os.makedirs(transformer_dest, exist_ok=True)
        
        # Save config and weights
        model.transformer.to("cpu")
        model.transformer.save_pretrained(transformer_dest, max_shard_size="10GB")
        
        # Copy other directories from original base model if they exist
        if model.tokenizer is not None:
            # We don't save tokenizer as it is read-only Qwen3-VL tokenizer
            pass

    def __save_safetensors(
            self,
            model: Krea2Model,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = model.transformer.state_dict()
        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def __save_internal(
            self,
            model: Krea2Model,
            destination: str,
    ):
        self.__save_diffusers(model, destination, None)

    def save(
            self,
            model: Krea2Model,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.DIFFUSERS:
                self.__save_diffusers(model, output_model_destination, dtype)
            case ModelFormat.SAFETENSORS:
                self.__save_safetensors(model, output_model_destination, dtype)
            case ModelFormat.INTERNAL:
                self.__save_internal(model, output_model_destination)
