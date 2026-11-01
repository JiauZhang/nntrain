import importlib

from conippets.config import Config


def load_class(path):
    module_name, separator, class_name = path.partition(":")
    if not separator or not module_name or not class_name:
        raise ValueError(f'model class must be "module:ClassName", got {path!r}')
    return getattr(importlib.import_module(module_name), class_name)


def build_model(spec, device=None):
    model_class = load_class(spec.class_path)
    if spec.pretrained:
        model = model_class.from_pretrained(spec.pretrained)
    elif spec.config:
        model = model_class(Config.from_json(spec.config))
    else:
        model = model_class()
    if device is not None:
        model.to(device)
    return model
