# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable

import torch
from monarch.actor import Actor, endpoint
from torchtitan.experiments.rl.types import Completion, ScoredCompletion

logger = logging.getLogger(__name__)


class Grader(Actor):
    """
    Scores generated completions using a reward function.

    Consumes pre-grouped completions (one group per prompt, each sharing
    the same ``expected_answer``) so that ``reward_fn`` can be called
    with its natural batched shape. The grader does not compute the
    grouping itself -- the controller provides it.

    Args:
        reward_fn: Callable ``(completions: list[str], expected_answer: str) -> torch.Tensor``
            returning one reward per input completion.
    """

    def __init__(
        self,
        reward_fn: Callable,
    ):
        self.reward_fn = reward_fn

        logger.info("Grader initialized")

    @endpoint
    async def score(
        self,
        completions_per_prompt: list[list[Completion]],
        expected_answers: list[str],
    ) -> list[ScoredCompletion]:
        """Score pre-grouped completions.

        Args:
            completions_per_prompt: One inner list per prompt; each
                inner list contains the completions sharing that prompt.
            expected_answers: One expected answer per prompt, parallel
                to ``completions_per_prompt``.

        Returns:
            Flat list of ScoredCompletions in the order produced by
            iterating ``completions_per_prompt`` then each inner list.
        """
        assert len(completions_per_prompt) == len(expected_answers), (
            f"expected parallel lists, got "
            f"{len(completions_per_prompt)} groups vs "
            f"{len(expected_answers)} expected_answers"
        )

        scored: list[ScoredCompletion] = []
        all_rewards: list[float] = []
        for group, expected in zip(completions_per_prompt, expected_answers):
            rewards = self.reward_fn([c.text for c in group], expected)
            for c, r in zip(group, rewards.tolist()):
                scored.append(ScoredCompletion(completion=c, reward=r))
                all_rewards.append(r)

        rewards_t = torch.tensor(all_rewards)
        logger.debug(
            f"Grader finished scoring {len(scored)} completions: "
            f"reward_mean={rewards_t.mean().item():.4f}, "
            f"reward_std={rewards_t.std().item():.4f}"
        )

        return scored
