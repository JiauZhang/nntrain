import json
from collections.abc import Mapping
from pathlib import Path

import torch
from safetensors.torch import save_file

MANIFEST_NAME = "nnm_weights.json"


def load_pretrained(model_class, path, **kwargs):
    return model_class.from_pretrained(path, **kwargs)


def save_pretrained(directory, model, config, *, module=None, task=None, capabilities=None, weight_name=None):
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config_name = f"{module}.json" if module else "config.json"
    weight_name = weight_name or (f"{module}.safetensors" if module else "model.safetensors")

    (directory / config_name).write_text(json.dumps(dict(config), indent=2, ensure_ascii=False) + "\n")
    save_file({key: value.contiguous() for key, value in model.state_dict().items()}, str(directory / weight_name))

    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"version": 1, "sources": []}
    manifest.setdefault("sources", [])
    if task:
        manifest["task"] = task
    if capabilities:
        manifest["capabilities"] = list(capabilities)

    if module:
        manifest.setdefault("modules", {})[module] = {"config": config_name, "weights": [weight_name]}
    else:
        manifest["config"] = config_name
        manifest["weights"] = [weight_name]

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def save_checkpoint(path, *, model, optimizer=None, scheduler=None, epoch=0, step=0, extra=None):
    state = {"model": model.state_dict(), "epoch": epoch, "step": step}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if extra is not None:
        state["extra"] = extra

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def restore_checkpoint(state, *, model, optimizer=None, scheduler=None):
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
