# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Literal

from torchtitan.config import ActivationCheckpointConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer


@dataclass(kw_only=True, slots=True)
class PagedStashActivationCheckpointConfig(ActivationCheckpointConfig):
    """Extended activation checkpoint config for paged stash experiments."""

    paged_stash_buffer_device: Literal["cuda", "cpu"] = "cuda"
    """Device for the paged stash buffer."""

    paged_stash_page_size: int = 64
    """Number of tokens per page in the paged stash buffer."""

    paged_stash_buffer_size_factor: float = 1.1
    """Factor to scale max_tokens for buffer over-provisioning."""

    paged_stash_separate_stream: bool = False
    """Enable async stream overlap for paged stash copy/pop operations."""


def to_paged_stash_config(
    base_config: Trainer.Config,
    model_registry: Callable[[str], ModelSpec],
):
    """Convert a base Trainer.Config to a PagedStashTrainer.Config.

    Copies all fields from the base config and replaces the model_spec with one
    from the paged_stash model_registry. The compile and activation_checkpoint
    fields are removed and left as defaults; callers should explicitly set them.
    """
    from torchtitan.experiments.cuda_graphable_moe.train import PagedStashTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    d["model_spec"] = model_registry(base_config.model_spec.flavor)
    d.pop("compile")
    d.pop("activation_checkpoint")

    return PagedStashTrainer.Config(**d)
