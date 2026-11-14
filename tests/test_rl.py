import copy

import pytest
import torch

from nnserve.engine import Engine, SamplingParams
from nntrain.config import ModelSpec, RLSpec, RolloutSpec, TrainConfig, TrainSpec
from nntrain.rl import GRPOTrainer, rollout_episode
from nntrain.trainer import Trainer
from tests.tiny import TinyCausalLM


def make_trainer(tmp_path, **train):
    train.setdefault("device", "cpu")
    config = TrainConfig(
        model=ModelSpec(class_path="tests.tiny:TinyCausalLM"),
        output=str(tmp_path / "runs"),
        train=TrainSpec(**train),
    )
    return Trainer(TinyCausalLM(), config, None)


def count_ones(episode):
    return float(episode.generated_ids.count(1))


def test_rollout_episode_runs_a_tool_turn():
    engine = Engine.from_model(TinyCausalLM())
    params = SamplingParams(max_new_tokens=2, eos_token_ids=())

    episode = rollout_episode(
        engine,
        [0],
        params,
        tool_fn=lambda turn, tokens: [2] if turn == 0 else None,
        max_turns=2,
    )

    assert episode.action_mask.count(False) == 1
    assert episode.action_mask.count(True) > 0
    assert episode.generated_ids


def test_grpo_fit_updates_weights_and_policy_version(tmp_path):
    torch.manual_seed(0)
    trainer = make_trainer(tmp_path, epochs=2, learning_rate=0.1)
    engine = Engine.from_model(trainer.model)
    grpo = GRPOTrainer(
        trainer,
        engine,
        count_ones,
        rollout=RolloutSpec(group_size=3, max_new_tokens=3, do_sample=True, seed=1),
        rl=RLSpec(),
    )

    metrics = grpo.fit([[0]], epochs=2)

    assert metrics["groups"] == 2
    assert metrics["global_step"] == 2
    assert metrics["policy_version"] == 2
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert engine.policy_version == 2


def test_grpo_with_kl_reference_runs(tmp_path):
    torch.manual_seed(0)
    trainer = make_trainer(tmp_path, epochs=1, learning_rate=0.1)
    reference = copy.deepcopy(trainer.model)
    engine = Engine.from_model(trainer.model)
    grpo = GRPOTrainer(
        trainer,
        engine,
        count_ones,
        rollout=RolloutSpec(group_size=2, max_new_tokens=2, do_sample=True, seed=2),
        rl=RLSpec(kl_coef=0.1),
        ref_model=reference,
    )

    metrics = grpo.fit([[0]], epochs=1)

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert metrics["policy_version"] == 1


def test_grpo_requires_a_reference_model_for_kl(tmp_path):
    trainer = make_trainer(tmp_path)
    engine = Engine.from_model(trainer.model)

    with pytest.raises(ValueError):
        GRPOTrainer(trainer, engine, count_ones, rl=RLSpec(kl_coef=0.1))


def test_rl_specs_validate():
    with pytest.raises(ValueError):
        RolloutSpec(group_size=0)
    with pytest.raises(ValueError):
        RolloutSpec(max_turns=0)
    with pytest.raises(ValueError):
        RLSpec(algorithm="ppo")
    with pytest.raises(ValueError):
        RLSpec(clip_range=1.0)
