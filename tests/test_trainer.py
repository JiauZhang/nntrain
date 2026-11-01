import pytest
import torch
from conippets.config import Config

from nntrain.config import ModelSpec, TrainConfig, TrainSpec
from nntrain.trainer import Trainer
from tests.tiny import TinyModel


def make_config(tmp_path, **train):
    train.setdefault("device", "cpu")
    return TrainConfig(
        model=ModelSpec(class_path="tests.tiny:TinyModel"),
        output=str(tmp_path / "runs"),
        train=TrainSpec(**train),
    )


def mse_loss(model, batch):
    inputs, targets = batch
    return torch.nn.functional.mse_loss(model(inputs), targets)


def test_fit_reduces_loss(tmp_path):
    torch.manual_seed(0)
    model = TinyModel(Config(in_features=1, out_features=1))
    config = make_config(tmp_path, epochs=200, learning_rate=0.1)
    inputs = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    targets = 2.0 * inputs + 1.0

    trainer = Trainer(model, config, mse_loss)
    with torch.no_grad():
        initial = float(mse_loss(model, (inputs, targets)))
    metrics = trainer.fit([(inputs, targets)])
    with torch.no_grad():
        final = float(mse_loss(model, (inputs, targets)))

    assert final < initial
    assert metrics["optimizer_steps"] == 200
    assert metrics["global_step"] == 200


def test_gradient_accumulation_matches_separate_steps(tmp_path):
    torch.manual_seed(0)
    base = TinyModel(Config(in_features=1, out_features=1))
    first = TinyModel(Config(in_features=1, out_features=1))
    second = TinyModel(Config(in_features=1, out_features=1))
    first.load_state_dict(base.state_dict())
    second.load_state_dict(base.state_dict())

    config = make_config(tmp_path, epochs=1, learning_rate=0.1)
    config_single = TrainConfig(
        model=config.model,
        output=config.output,
        train=TrainSpec(epochs=1, learning_rate=0.1, grad_accumulation=1, device="cpu"),
    )
    config_accumulated = TrainConfig(
        model=config.model,
        output=config.output,
        train=TrainSpec(epochs=1, learning_rate=0.1, grad_accumulation=2, device="cpu"),
    )

    batch_a = (torch.tensor([[1.0]]), torch.tensor([[3.0]]))
    batch_b = (torch.tensor([[2.0]]), torch.tensor([[5.0]]))
    Trainer(first, config_single, mse_loss, optimizer=torch.optim.SGD(first.parameters(), lr=0.1)).fit([batch_a, batch_b])
    Trainer(second, config_accumulated, mse_loss, optimizer=torch.optim.SGD(second.parameters(), lr=0.1)).fit(
        [batch_a, batch_a, batch_b, batch_b]
    )

    torch.testing.assert_close(first.linear.weight, second.linear.weight, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(first.linear.bias, second.linear.bias, atol=1e-6, rtol=1e-6)


def test_invalid_train_spec(tmp_path):
    with pytest.raises(ValueError):
        TrainSpec(grad_accumulation=0)
    with pytest.raises(ValueError):
        TrainSpec(amp="fp8")
    with pytest.raises(ValueError):
        TrainSpec(device="tpu")
