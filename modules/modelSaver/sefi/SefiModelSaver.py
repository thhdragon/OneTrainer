import os.path
from pathlib import Path

from modules.model.SefiModel import SefiModel
from modules.modelSaver.mixin.DtypeModelSaverMixin import DtypeModelSaverMixin
from modules.util.convert_util import convert
from modules.util.enum.ModelFormat import ModelFormat

import torch
from safetensors.torch import save_file


class SefiModelSaver(
    DtypeModelSaverMixin,
):
    def __init__(self):
        super().__init__()

    def __save_safetensors(
            self,
            model: SefiModel,
            destination: str,
            dtype: torch.dtype | None,
    ):
        state_dict = model.transformer.state_dict()
        state_dict = convert(state_dict, model.checkpoint_diffusers_to_original())

        save_state_dict = self._convert_state_dict_dtype(state_dict, dtype)
        self._convert_state_dict_to_contiguous(save_state_dict)

        os.makedirs(Path(destination).parent.absolute(), exist_ok=True)
        save_file(save_state_dict, destination, self._create_safetensors_header(model, save_state_dict))

    def save(
            self,
            model: SefiModel,
            output_model_format: ModelFormat,
            output_model_destination: str,
            dtype: torch.dtype | None,
    ):
        match output_model_format:
            case ModelFormat.LEGACY_SAFETENSORS | ModelFormat.ORIGINAL_TRANSFORMER:
                self.__save_safetensors(model, output_model_destination, dtype)
            case _:
                raise NotImplementedError(f"Unsupported output format: {output_model_format}")
