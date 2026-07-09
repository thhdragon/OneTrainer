from modules.model.SefiModel import SefiModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.modelSaver.flux.FluxLoRASaver import FluxLoRASaver
from modules.util.enum.ModelType import ModelType

SefiLoRAModelSaver = make_lora_model_saver(
    ModelType.SEFI,
    model_class=SefiModel,
    lora_saver_class=FluxLoRASaver,
    embedding_saver_class=None,
)
