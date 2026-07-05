from modules.model.SefiModel import SefiModel
from modules.modelSaver.sefi.SefiModelSaver import SefiModelSaver
from modules.modelSaver.GenericFineTuneModelSaver import make_fine_tune_model_saver
from modules.util.enum.ModelType import ModelType

SefiFineTuneModelSaver = make_fine_tune_model_saver(
    ModelType.SEFI,
    model_class=SefiModel,
    model_saver_class=SefiModelSaver,
    embedding_saver_class=None,
)
