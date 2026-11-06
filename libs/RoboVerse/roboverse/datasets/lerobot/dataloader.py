# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the CC BY-NC 4.0 license [see LICENSE for details].

# Suppress unnnecessary warnings
import logging
import warnings
from functools import cache

import torch
from einops import rearrange
from roboverse import constants as c
from torch.utils.data import Dataset


logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


@cache
def get_lerobot_metadata(repo_id):
    """
    Get the metadata for a LeRobot dataset
    :param repo_id: (str) Repository ID from huggingface to load the dataset
    :return: (dict) Metadata for the dataset
    """
    return LeRobotDatasetMetadata(repo_id=repo_id)


def get_final_le_cam_list_rv_cam_list(metadata, le_cam_list, rv_cam_list):
    """
    This is a helper function that is used to get the final le_cam_list and rv_cam_list used in the dataset.
    It checks if the camera names are valid and returns the final le_cam_list and rv_cam_list.
    It works even when le_cam_list and rv_cam_list are None. The le_cam_list is one to one mapped to rv_cam_list.
    It is used in LeRobotRV.__init__ and le_sample_to_rv_sample.
    :param metadata:
    :param le_cam_list: (list) List of le robot camera names to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param rv_cam_list: (list) Corresponding list of camera names in roboverse to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    """
    # Convert to tuple for caching if needed, convert back to list for compatibility
    if le_cam_list is None:
        le_cam_list = list(metadata.camera_keys)
    else:
        le_cam_list = (
            list(le_cam_list) if not isinstance(le_cam_list, list) else le_cam_list
        )
        print(f"le_cam_list: {le_cam_list}")
        print(f"metadata.camera_keys: {metadata.camera_keys}")
        for _cam in le_cam_list:
            assert _cam in metadata.camera_keys, f"Camera {_cam} not found in metadata"

    if rv_cam_list is None:
        rv_cam_list = c.CAMERA_NAMES[: len(le_cam_list)]
    else:
        rv_cam_list = (
            list(rv_cam_list) if not isinstance(rv_cam_list, list) else rv_cam_list
        )
        for _cam in rv_cam_list:
            assert (
                _cam in c.CAMERA_NAMES
            ), f"Camera {_cam} not found in roboverse constants"

    assert len(le_cam_list) == len(
        rv_cam_list
    ), "Length of le_cam_list and rv_cam_list should be same"

    return le_cam_list, rv_cam_list


def le_sample_to_rv_sample(
    sample,
    history,
    horizon,
    repo_id,
    le_cam_list,
    rv_cam_list,
    action_key,
    state_key,
    add_ori_act=True,
    add_out_ori_act=True,
):
    """
    Convert a sample from LeRobot dataset to a sample in RV dataset. Should use the exact same arguments as LeRobotRV.__init__.
    outputs in Numpy.
    :param sample: (dict) A sample from LeRobot dataset
    :param history: (int) Number of history frames
    :param horizon: (int) Number of future action
    :param repo_id: (str) Repository ID from huggingface to load the dataset
    :param le_cam_list: (list) List of le robot camera names to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param rv_cam_list: (list) Corresponding list of camera names in roboverse to be used as input. Similar to the one provided in LeRobotRV.__init__. Could be None.
    :param action_key: (str) Key for action in the dataset
    :param state_key: (str) Key for state in the dataset
    :param add_ori_act: (bool) Whether to add the original action to the sample
    :param add_out_ori_act: (bool) Whether to add the original action to the sample
    :return: (dict) A sample in RV dataset
    """
    rv_sample = {}
    metadata = get_lerobot_metadata(repo_id)
    le_cam_list, _ = get_final_le_cam_list_rv_cam_list(
        metadata, le_cam_list, rv_cam_list
    )

    rgb = [sample[x] * 255 for x in le_cam_list]
    if history > 1:
        rgb = [rearrange(x, "hi c h w -> hi 1 h w c") for x in rgb]
    else:
        if rgb[0].ndim == 3:
            rgb = [rearrange(x, "c h w -> 1 1 h w c") for x in rgb]
        else:
            # for libero, we find that the rgb is of shape (1, 3, 224, 224)
            # not ideal but we can still use it
            rgb = [rearrange(x, "1 c h w -> 1 1 h w c") for x in rgb]
    rv_sample["rgb"] = torch.cat(rgb, dim=1).numpy()
    rv_sample["instr"] = sample["task"]
    rv_sample["proprio"] = sample[state_key]
    if add_ori_act:
        rv_sample["ori_act"] = sample[action_key][:history]
    if add_out_ori_act:
        rv_sample["out_ori_act"] = sample[action_key][history:]

    return rv_sample


class LeRobotRV(Dataset):
    """
    RV dataset that wraps on top of any LeRobot dataset
    Based on information provided here: https://github.com/huggingface/lerobot/blob/main/examples/1_load_lerobot_dataset.py
    """

    def __init__(
        self,
        history,
        horizon,
        repo_id,
        le_cam_list,
        rv_cam_list,
        action_key="action",
        state_key="observation.state",
        fps=-1,
    ):
        """
        :param history: (int) Number of history frames
        :param horizon: (int) Number of future action
        :param repo_id: (str) Repository ID from huggingface to load the dataset.
        :param le_cam_list: (list) List of le robot camera names to be used as input.
            If None, all cameras are used. The ID should match the keys in
            metadata.camera_keys
        :param rv_cam_list: (list) Corresponding list of camera names in
            roboverse to be used as input. The length of this list should match the
            length of le_cam_list. If None, each le_camera is mapped to a
            corresponding camera in CAMERA_NAMES in roboverse constants
        :param fps: (int) Frame rate used for the delta timestamps; -1 uses the dataset fps.
        """
        self.repo_id = repo_id
        self.history = history
        self.horizon = horizon

        # get metadata
        metadata = get_lerobot_metadata(self.repo_id)

        # set action and state keys
        assert (
            action_key in metadata.features
        ), f"Action key '{action_key}' not found in metadata. Available metadata keys: {metadata}"
        assert (
            state_key in metadata.features
        ), f"Observation key '{state_key}' not found in metadata. Available metadata keys: {metadata}"
        self.action_key = action_key
        self.state_key = state_key

        # set other attributes
        self.threed_compatible = False
        self.ee_compatible = False
        self.has_original_action = True
        self.original_action_dim = metadata.features[self.action_key]["shape"][0]
        self.has_proprio = True
        self.proprio_dim = metadata.features[self.state_key]["shape"][0]

        # set cam_list that is required for RoboVerse
        self.le_cam_list = le_cam_list
        self.rv_cam_list = rv_cam_list
        # _le_cam_list is the final le_cam_list that is used in the dataset
        # it is not the same as le_cam_list, it does the processing in get_final_le_cam_list_rv_cam_list
        _le_cam_list, _rv_cam_list = get_final_le_cam_list_rv_cam_list(
            metadata, le_cam_list, rv_cam_list
        )
        print(f"LeRobot and RoboVerse Camera Mapping: {_le_cam_list} -> {_rv_cam_list}")
        self.cam_list = _rv_cam_list

        # get the LeRobotDataset
        if fps == -1:
            fps = metadata.fps
        delta_timestamps = {
            # loads proprio from -history+1 to 0
            self.state_key: [-x / fps for x in range(history - 1, -1, -1)],
            # loads action from timesteps -history to horizon-1
            self.action_key: [
                x / fps for x in range(-history, horizon)
            ],  # for both past and future actions
        }
        print(f"Delta Timestamps: {delta_timestamps}")
        for cam in _le_cam_list:
            # loads camera frames from -history+1 to 0
            delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]

        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            delta_timestamps=delta_timestamps,
        )

        # min / max of the actions, used to discretize them
        act_stats = self.dataset.meta.stats[self.action_key]
        del act_stats["mean"]
        del act_stats["std"]
        self.stats = {
            "out_ori_act": act_stats,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset.__getitem__(idx)
        rv_sample = le_sample_to_rv_sample(
            sample,
            self.history,
            self.horizon,
            self.repo_id,
            self.le_cam_list,
            self.rv_cam_list,
            self.action_key,
            self.state_key,
        )
        return rv_sample

