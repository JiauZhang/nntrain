import json

import torch
from conippets.config import Config

from nntrain.checkpoint import load_checkpoint, restore_checkpoint, save_pretrained
from nntrain.config import ModelSpec, TrainConfig, TrainSpec
from nntrain.model import build_model
from nntrain.trainer import Trainer
from tests.tiny import DecoderModel, EncoderModel, TinyModel


def mse_loss(model, batch):
    inputs, targets = batch
    return torch.nn.functional.mse_loss(model(inputs), targets)


def make_trainer(tmp_path, model):
    config = TrainConfig(
        model=ModelSpec(class_path="tests.tiny:TinyModel"),
        output=str(tmp_path / "runs"),
        train=TrainSpec(epochs=1, learning_rate=0.1, device="cpu"),
        task="chat",
        capabilities=["vision"],
    )
    return Trainer(model, config, mse_loss)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    model = TinyModel(Config(in_features=1, out_features=1))
    trainer = make_trainer(tmp_path, model)
    trainer.fit([(torch.tensor([[1.0]]), torch.tensor([[2.0]]))])

    path = tmp_path / "trainer.pt"
    trainer.save_checkpoint(path)

    restored = TinyModel(Config(in_features=1, out_features=1))
    resumed = make_trainer(tmp_path, restored)
    state = resumed.load_checkpoint(path)

    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert state["step"] == trainer.global_step
    assert resumed.global_step == trainer.global_step


def test_restore_checkpoint_requires_matching_optimizer(tmp_path):
    model = TinyModel(Config(in_features=1, out_features=1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)

    state = load_checkpoint(path)
    restored = TinyModel(Config(in_features=1, out_features=1))
    restore_checkpoint(state, model=restored, optimizer=optimizer)

    assert state["optimizer"]["param_groups"][0]["lr"] == 0.1


def test_save_pretrained_is_loadable_by_nnm(tmp_path):
    model = TinyModel(Config(in_features=2, out_features=3))
    save_pretrained(tmp_path, model, {"in_features": 2, "out_features": 3}, task="chat", capabilities=["vision"])

    manifest = json.loads((tmp_path / "nnm_weights.json").read_text())
    assert manifest["version"] == 1
    assert manifest["sources"] == []
    assert manifest["weights"] == ["model.safetensors"]
    assert manifest["task"] == "chat"
    assert manifest["capabilities"] == ["vision"]

    loaded = TinyModel.from_pretrained(tmp_path)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_save_pretrained_merges_modules(tmp_path):
    model = TinyModel(Config(in_features=2, out_features=3))
    config = {"in_features": 2, "out_features": 3}
    save_pretrained(tmp_path, model, config, module="encoder", task="chat")
    save_pretrained(tmp_path, model, config, module="decoder", task="chat")

    manifest = json.loads((tmp_path / "nnm_weights.json").read_text())
    assert set(manifest["modules"]) == {"encoder", "decoder"}
    assert manifest["modules"]["encoder"]["config"] == "encoder.json"
    assert manifest["modules"]["decoder"]["weights"] == ["decoder.safetensors"]
    assert manifest["task"] == "chat"

    loaded_encoder = EncoderModel.from_pretrained(tmp_path)
    loaded_decoder = DecoderModel.from_pretrained(tmp_path)
    assert loaded_encoder.linear.weight.shape == (3, 2)
    assert loaded_decoder.linear.bias.shape == (3,)


def test_build_model_from_config(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"in_features": 2, "out_features": 4}))

    model = build_model(ModelSpec(class_path="tests.tiny:TinyModel", config=str(config_path)))

    assert model.linear.in_features == 2
    assert model.linear.out_features == 4
