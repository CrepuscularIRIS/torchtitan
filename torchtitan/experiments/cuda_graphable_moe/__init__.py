# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.graph_trainer.passes import AVAILABLE_JOINT_PASSES

from .paged_stash_graph_pass import apply_paged_sac_pass, apply_sac_grouped_mm_pass

AVAILABLE_JOINT_PASSES["apply_paged_sac"] = apply_paged_sac_pass
AVAILABLE_JOINT_PASSES["apply_sac_grouped_mm"] = apply_sac_grouped_mm_pass
