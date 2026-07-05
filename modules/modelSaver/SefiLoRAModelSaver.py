from modules.model.SefiModel import SefiModel
from modules.modelSaver.GenericLoRAModelSaver import make_lora_model_saver
from modules.util.enum.ModelType import ModelType

SefiLoRAModelSaver = make_lora_model_saver(
    ModelType.SEFI,
    model_class=SefiModel,
    embedding_saver_class=None,
)
