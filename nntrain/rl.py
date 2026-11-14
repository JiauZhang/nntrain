from dataclasses import dataclass
from typing import Callable

import torch

from lossers.rl import grpo_loss, kl_penalty
from nnserve.engine import Engine, SamplingParams

from .config import RLSpec, RolloutSpec
from .trainer import Trainer


@dataclass
class Episode:
    prompt_ids: list[int]
    token_ids: list[int]
    action_mask: list[bool]
    old_logprobs: list[float]
    reward: float = 0.0

    @property
    def generated_ids(self):
        return [token for token, action in zip(self.token_ids, self.action_mask) if action]


def rollout_episode(
    engine: Engine,
    prompt_ids,
    params: SamplingParams,
    tool_fn: Callable[[int, list[int]], list[int] | None] | None = None,
    max_turns: int = 1,
) -> Episode:
    session = engine.create_session()
    token_ids = []
    action_mask = []
    old_logprobs = []
    try:
        for turn in range(max_turns):
            result = engine.generate(session, prompt_ids if turn == 0 else None, params)
            token_ids.extend(result.tokens)
            action_mask.extend([True] * len(result.tokens))
            old_logprobs.extend(result.logprobs)
            if tool_fn is None or turn + 1 == max_turns:
                break
            observation = tool_fn(turn, result.tokens)
            if observation is None:
                break
            engine.prefill(session, observation)
            token_ids.extend(observation)
            action_mask.extend([False] * len(observation))
            old_logprobs.extend([0.0] * len(observation))
    finally:
        engine.close_session(session)
    return Episode(list(prompt_ids), token_ids, action_mask, old_logprobs)


class GRPOTrainer:
    def __init__(
        self,
        trainer: Trainer,
        engine: Engine,
        reward_fn: Callable[[Episode], float],
        rollout: RolloutSpec | None = None,
        rl: RLSpec | None = None,
        ref_model=None,
        tool_fn: Callable[[int, list[int]], list[int] | None] | None = None,
    ):
        self.trainer = trainer
        self.engine = engine
        self.reward_fn = reward_fn
        self.rollout = rollout or RolloutSpec()
        self.rl = rl or RLSpec()
        self.tool_fn = tool_fn
        self.ref_model = ref_model
        if self.rl.kl_coef > 0:
            if ref_model is None:
                raise ValueError("ref_model is required when kl_coef is positive")
            self.ref_model = ref_model.to(trainer.device).eval()

    def fit(self, prompts, epochs=None):
        prompts = [list(prompt) for prompt in prompts]
        if not prompts:
            raise ValueError("prompts must not be empty")
        epochs = epochs or self.trainer.config.train.epochs
        total_loss = 0.0
        total_reward = 0.0
        groups = 0

        for _ in range(epochs):
            for prompt in prompts:
                episodes = self._rollout_group(prompt)
                rewards = torch.tensor([episode.reward for episode in episodes], device=self.trainer.device)
                advantages = self._advantages(rewards)
                batch = self._collate(prompt, episodes, advantages)

                self.trainer.model.train()
                with self.trainer.autocast:
                    logprobs, mask = self._policy_logprobs(batch["input_ids"], batch["loss_mask"])
                    loss = grpo_loss(
                        logprobs,
                        batch["old_logprobs"],
                        batch["advantages"],
                        mask=mask,
                        clip_range=self.rl.clip_range,
                    )
                    if self.ref_model is not None:
                        with torch.no_grad():
                            ref_logprobs, _ = self._logprobs_for(self.ref_model, batch["input_ids"], batch["loss_mask"])
                        loss = loss + self.rl.kl_coef * kl_penalty(logprobs, ref_logprobs, mask)

                self.trainer.backward(loss, 1)
                self.trainer.step()
                total_loss += float(loss.detach())
                total_reward += float(rewards.mean())
                groups += 1
                self._sync_engine()

        return {
            "loss": total_loss / groups,
            "reward": total_reward / groups,
            "groups": groups,
            "global_step": self.trainer.global_step,
            "policy_version": self.engine.policy_version,
        }

    def _rollout_group(self, prompt):
        episodes = []
        self.trainer.model.eval()
        for index in range(self.rollout.group_size):
            seed = None if self.rollout.seed is None else self.rollout.seed + index
            params = SamplingParams(
                max_new_tokens=self.rollout.max_new_tokens,
                do_sample=self.rollout.do_sample,
                temperature=self.rollout.temperature,
                top_k=self.rollout.top_k,
                top_p=self.rollout.top_p,
                seed=seed,
            )
            episode = rollout_episode(
                self.engine,
                prompt,
                params,
                tool_fn=self.tool_fn,
                max_turns=self.rollout.max_turns,
            )
            episode.reward = float(self.reward_fn(episode))
            episodes.append(episode)
        return episodes

    def _advantages(self, rewards):
        if not self.rl.normalize_advantages:
            return rewards
        if rewards.numel() == 1:
            return rewards - rewards.mean()
        return (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)

    def _collate(self, prompt, episodes, advantages):
        lengths = [len(prompt) + len(episode.token_ids) for episode in episodes]
        max_length = max(lengths)
        input_ids = torch.zeros(len(episodes), max_length, dtype=torch.long, device=self.trainer.device)
        loss_mask = torch.zeros(len(episodes), max_length, dtype=torch.bool, device=self.trainer.device)
        old_logprobs = torch.zeros(len(episodes), max_length, device=self.trainer.device)
        for index, episode in enumerate(episodes):
            start = len(prompt)
            end = start + len(episode.token_ids)
            input_ids[index, :start] = torch.tensor(prompt, dtype=torch.long, device=self.trainer.device)
            input_ids[index, start:end] = torch.tensor(episode.token_ids, dtype=torch.long, device=self.trainer.device)
            loss_mask[index, start:end] = torch.tensor(episode.action_mask, dtype=torch.bool, device=self.trainer.device)
            old_logprobs[index, start:end] = torch.tensor(episode.old_logprobs, device=self.trainer.device)
        return {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
            "old_logprobs": old_logprobs[:, 1:],
            "advantages": advantages.unsqueeze(-1),
        }

    def _policy_logprobs(self, input_ids, loss_mask):
        return self._logprobs_for(self.trainer.model, input_ids, loss_mask)

    def _logprobs_for(self, model, input_ids, loss_mask):
        logits = model(input_ids)
        log_probs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = input_ids[:, 1:].unsqueeze(-1)
        return log_probs.gather(-1, targets).squeeze(-1), loss_mask[:, 1:]

    def _sync_engine(self):
        version = self.engine.policy_version + 1
        if self.engine.model is self.trainer.model:
            self.engine.update_weights(policy_version=version)
        else:
            self.engine.update_weights(self.trainer.model.state_dict(), policy_version=version)
