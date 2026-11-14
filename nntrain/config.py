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
class RolloutSpec:
    group_size: int = 4
    max_new_tokens: int = 32
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    max_turns: int = 1

    def __post_init__(self):
        if self.group_size < 1:
            raise ValueError("group_size must be at least 1")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass
class RLSpec:
    algorithm: str = "grpo"
    clip_range: float = 0.2
    kl_coef: float = 0.0
    normalize_advantages: bool = True

    def __post_init__(self):
        if self.algorithm != "grpo":
            raise ValueError("only the grpo algorithm is supported")
        if not 0 <= self.clip_range < 1:
            raise ValueError("clip_range must be in [0, 1)")
        if self.kl_coef < 0:
            raise ValueError("kl_coef must be non-negative")


@dataclass
class TrainConfig:
    model: ModelSpec
    output: str
    train: TrainSpec = field(default_factory=TrainSpec)
    rollout: RolloutSpec | None = None
    rl: RLSpec | None = None
    task: str | None = None
    capabilities: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, path):
        data = json.loads(Path(path).read_text())
        model = ModelSpec(**data["model"])
        train = TrainSpec(**data.get("train", {}))
        rollout = RolloutSpec(**data["rollout"]) if "rollout" in data else None
        rl = RLSpec(**data["rl"]) if "rl" in data else None
        return cls(
            model=model,
            output=data["output"],
            train=train,
            rollout=rollout,
            rl=rl,
            task=data.get("task"),
            capabilities=data.get("capabilities", []),
        )
