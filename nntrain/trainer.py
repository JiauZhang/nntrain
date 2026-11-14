import contextlib

import torch

from nntrain.checkpoint import load_checkpoint as load_training_checkpoint
from nntrain.checkpoint import restore_checkpoint
from nntrain.checkpoint import save_checkpoint as save_training_checkpoint
from nntrain.checkpoint import save_pretrained as save_pretrained_model
from nntrain.config import TrainConfig


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available")
    return device


def resolve_amp(name, device_type):
    if name is None:
        return None
    if name == "fp16":
        if device_type != "cuda":
            raise ValueError("fp16 autocast requires CUDA")
        return torch.float16
    if device_type not in ("cuda", "cpu"):
        raise ValueError("bf16 autocast requires CUDA or CPU")
    return torch.bfloat16


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(to_device(item, device) for item in value)
    return value


class Trainer:
    def __init__(self, model, config: TrainConfig, loss_fn, optimizer=None, scheduler=None):
        self.config = config
        self.device = resolve_device(config.train.device)
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = scheduler
        self.amp_dtype = resolve_amp(config.train.amp, self.device.type)
        fp16 = self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16) if fp16 else None
        self.epoch = 0
        self.global_step = 0

    @property
    def autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def fit(self, batches, epochs=None):
        epochs = epochs or self.config.train.epochs
        accumulation = self.config.train.grad_accumulation

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        micro_steps = 0
        optimizer_steps = 0
        pending = 0

        for _ in range(epochs):
            for batch in batches:
                batch = to_device(batch, self.device)
                with self.autocast:
                    loss = self.loss_fn(self.model, batch)
                self.backward(loss, accumulation)

                total_loss += float(loss.detach())
                micro_steps += 1
                pending += 1
                if pending == accumulation:
                    self.step()
                    pending = 0
                    optimizer_steps += 1
            if pending:
                self.step()
                pending = 0
                optimizer_steps += 1
            self.epoch += 1

        return {
            "loss": total_loss / micro_steps if micro_steps else 0.0,
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "global_step": self.global_step,
        }

    def backward(self, loss, accumulation):
        scaled = loss / accumulation
        if self.scaler is not None:
            self.scaler.scale(scaled).backward()
        else:
            scaled.backward()

    def step(self):
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        if self.config.train.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self.global_step += 1

    def save_checkpoint(self, path, extra=None):
        save_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.global_step,
            extra=extra,
        )

    def load_checkpoint(self, path):
        state = load_training_checkpoint(path, map_location=self.device)
        restore_checkpoint(state, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler)
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("step", 0))
        return state

    def save_pretrained(self, directory, config, **kwargs):
        save_pretrained_model(
            directory,
            self.model,
            config,
            task=self.config.task,
            capabilities=self.config.capabilities,
            **kwargs,
        )
