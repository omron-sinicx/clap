# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].


def get_unifier_stats(dataset, cfg, stat_keys):
    """
    Get the stats of the unifier.

    Only single-dataset unifiers whose dataset already provides the stats (as the
    LeRobot datasets do, from their metadata) are supported.
    :param dataset: The dataset to get the stats of.
    :param cfg: The config of the unifier (unused, kept for API compatibility).
    :param stat_keys: The keys to get the stats of.
    :return: A dictionary of the stats.
    """
    if len(dataset.datasets) == 1:
        _dataset = list(dataset.datasets.values())[0]
        if hasattr(_dataset, "stats") and all(key in _dataset.stats for key in stat_keys):
            return {key: _dataset.stats[key] for key in stat_keys}
    raise NotImplementedError(
        f"stats for {stat_keys} must be provided by a single dataset; "
        "computing them by iterating over the data is not supported"
    )
