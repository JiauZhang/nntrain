import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    class_path: str
    config: str | None = None
    pretrained: str | None = None


@dataclass
class TrainSpec:
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_accumulation: int = 1
    max_grad_norm: float | None = 1.0
    amp: str | None = None
    device: str = "auto"
    seed: int = 0

    def __post_init__(self):
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.grad_accumulation < 1:
            raise ValueError("grad_accumulation must be at least 1")
        if self.amp not in (None, "fp16", "bf16"):
            raise ValueError("amp must be fp16, bf16 or null")
        if self.device not in ("auto", "cpu", "cuda", "mps"):
            raise ValueError("device must be auto, cpu, cuda or mps")


@dataclass
class TrainConfig:
    model: ModelSpec
    output: str
    train: TrainSpec = field(default_factory=TrainSpec)
    task: str | None = None
    capabilities: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        model = ModelSpec(**data["model"])
        train = TrainSpec(**data.get("train", {}))
        return cls(
            model=model,
            output=data["output"],
            train=train,
            task=data.get("task"),
            capabilities=data.get("capabilities", []),
        )
