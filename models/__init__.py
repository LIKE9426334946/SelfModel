from .model_01 import Model01
from .model_02 import Model02
from .model_03 import Model03
from .model_04 import Model04
from .model_05 import Model05
from .model_06 import Model06
from .model_07 import Model07

MODEL_REGISTRY = {
    "model_01": Model01,
    "model_02": Model02,
    "model_03": Model03,
    "model_04": Model04,
    "model_05": Model05,
    "model_06": Model06,
    "model_07": Model07,
}


def build_model(model_config, data_config):
    model_name = model_config["name"]
    model_class = MODEL_REGISTRY[model_name]
    kwargs = {
        "in_channels": data_config["in_channels"],
        "out_channels": (
            1 if data_config["mode"] == "binary" else data_config["num_classes"]
        ),
    }

    if model_name == "model_03":
        kwargs["image_size"] = tuple(data_config["image_size"])

    return model_class(**kwargs)
