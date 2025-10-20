# file copied from sparsh/tactile_ssl/data/vision_based_force_probe.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from omegaconf import DictConfig, OmegaConf
import numpy as np

import torch
from torch.utils import data
import torchvision.transforms.functional as TF
from torchvision import transforms
import matplotlib.pyplot as plt
import wandb

from sparsh.tactile_ssl.data.digit.utils import (
    load_dataset_forces,
    load_sample_from_buf,
    get_resize_transform,
    load_bin_image,
)
from sparsh.tactile_ssl.utils.logging import get_pylogger

log = get_pylogger()

DEBUG = False

# config comes from sparsh/config/data/digit_force.yaml
config = OmegaConf.create({
    "remove_bg": True,
    "out_format": "concat_ch_img",
    "num_frames": 2,
    "frame_stride": 5,
    "sensor": "digit",
    "path_dataset": "data/digit-force-estimation/",
    "transforms": {
        "resize": [224, 224],
    },
    "sphere_max_abs_forceXYZ": [4.0, 4.0, 5.0], # N
    "sharp_max_abs_forceXYZ": [4.0, 4.0, 5.0], # N
    "hex_max_abs_forceXYZ": [4.0, 4.0, 5.0], # N
    "max_abs_forceXYZ": [4.0, 4.0, 5.0], # N (if need to assign different values for each dataset, use sphere_max_abs_forceXYZ, sharp_max_abs_forceXYZ, hex_max_abs_forceXYZ)
    "max_delta_forceXYZ": [0.25, 0.25, 0.15], #N
    "slip_horizon": 0,
    "train_data_budget": 0.01, # fraction of train dataset to use (between 0 and 1)
    "list_datasets": [
      "sphere/batch_1",
      "sharp/batch_1",
      "flat/batch_2",
    ],
    "list_datasets_test": [
      "flat/batch_1",
    ]
})

SLIP_LABELS = {0: "no_slip", 1: "slip"}


class VisionForceSlipDataset(data.Dataset):
    def __init__(
        self,
        config: DictConfig,
        dataset_name: str,
    ):

        super().__init__()
        self.config = config
        self.dataset_name = dataset_name
        self.slip_horizon = config.slip_horizon

        self.remove_bg = (
            self.config.remove_bg if hasattr(self.config, "remove_bg") else False
        )
        self.out_format = self.config.out_format  # if output video
        assert self.out_format in [
            "video",
            "concat_ch_img",
            "single_image",
        ], ValueError(
            "out_format should be 'video' or 'concat_ch_img' or 'single_image'"
        )

        frame_stride = self.config.frame_stride
        self.num_frames = (
            1 if self.out_format == "single_image" else self.config.num_frames
        )
        self.frames_concat_idx = np.arange(
            0, self.num_frames * frame_stride, frame_stride
        )

        # load dataset
        self.frames, dataset_force_slip = load_dataset_forces(
            self.config, dataset_name, config.sensor
        )
        # force and other labels
        self.in_contact = dataset_force_slip["in_contact"]
        self.trajectories = dataset_force_slip["trajectories"]
        self.idx2traj, self.traj2idx, self.slip_labels = self.get_map_idx2traj()

        # transforms
        self.img_sz = self.config.transforms.resize
        if self.out_format == "concat_ch_img" and self.num_frames == 2:
            self.transform_resize = get_resize_transform([int(self.img_sz[0]/2), self.img_sz[1]])
        else:
            self.transform_resize = get_resize_transform(self.img_sz)

        # background
        self.bg = None
        if self.remove_bg:
            idx_bg = np.where(self.in_contact == 0)[0][0]
            self.bg = load_bin_image(self.frames[idx_bg])

        # force scale
        self.max_abs_forceXYZ = self.config.max_abs_forceXYZ
        self.max_delta_forceXYZ = self.config.max_delta_forceXYZ

    def get_map_idx2traj(self):
        slip_labels_all = []
        idx2traj = {}
        traj2idx = {}
        idx = -1
        for trajectory_i in self.trajectories.keys():
            traj2idx[trajectory_i] = []
            t_idxs = self.trajectories[trajectory_i]["indexes"][5:]
            for sample in range(len(t_idxs)):
                idx += 1
                traj2idx[trajectory_i].append(idx)
                idx2traj[idx] = {}
                idx2traj[idx]["trajectory"] = trajectory_i
                idx2traj[idx]["sample"] = sample
                slip_label = self._get_slip_labels(trajectory_i, sample)
                idx2traj[idx]["slip_horizon_labels"] = slip_label
                slip_labels_all.append(0 if slip_label.sum() == 0 else 1)
        slip_labels_all = np.array(slip_labels_all)
        return idx2traj, traj2idx, slip_labels_all

    def _get_slip_labels(self, trajectory_i, sample):
        slip_horizon_labels = []
        slip_labels = self.trajectories[trajectory_i]["slip_label"]
        len_trajectory = len(slip_labels)

        t = sample + np.arange(0, self.slip_horizon + 1)
        t = np.clip(t, 0, len_trajectory - 1)
        slip_horizon_labels = slip_labels[t]
        return slip_horizon_labels.astype(int)

    def __len__(self):
        return len(self.idx2traj)

    def _len_trajectory(self, idx_trajectory):
        return len(self.trajectories[idx_trajectory])

    def __getitem__(self, idx):
        idx_trajectory = self.idx2traj[int(idx)]["trajectory"]
        idx_sample = self.idx2traj[int(idx)]["sample"]
        label, abs_force, delta_force = self._get_force_slip_labels(idx, idx_trajectory, idx_sample)

        try:
            inputs = {}
            inputs["image"] = self._get_tactile_images(idx_trajectory, idx_sample)
            inputs["force"] = torch.tensor(abs_force).float()
            inputs["delta_force"] = torch.tensor(delta_force).float()
            inputs["slip_label"] = torch.tensor(label)
            inputs["slip_category_label"] = SLIP_LABELS[label]
            inputs["force_scale"] = torch.tensor(self.max_abs_forceXYZ, dtype=torch.float32)
            inputs["delta_force_scale"] = torch.tensor(self.max_delta_forceXYZ, dtype=torch.float32)
            return inputs

        except Exception as e:
            log.warn(f"Error in loading sample {idx}: {e}. Resampling")
            self.__getitem__(np.random.randint(0, self.__len__()))

    def _get_force_slip_labels(self, idx, idx_trajectory, idx_sample):
        '''
        We use the slip horizon labels to debounce fast switches in slip events, which is likely noise
        if there is atleast one slip event in the window the whole window is considered as slipping
        '''
        slip_horizon_labels = self.idx2traj[int(idx)]["slip_horizon_labels"]
        label = 0 if slip_horizon_labels.sum() == 0 else 1

        len_trajectory = len(self.trajectories[idx_trajectory]["indexes"])
        idx_sample_prev = np.clip(idx_sample - self.frames_concat_idx[-1], 0, len_trajectory - 1)
        f_curr = self.trajectories[idx_trajectory]["forces"][idx_sample]
        f_prev = self.trajectories[idx_trajectory]["forces"][idx_sample_prev]
        delta_forces = f_curr - f_prev
        delta_forces = delta_forces / self.max_delta_forceXYZ
        delta_forces = np.clip(delta_forces, -1.0, 1.0)

        abs_forces = self.trajectories[idx_trajectory]["forces"][idx_sample]
        abs_forces = abs_forces / self.max_abs_forceXYZ
        abs_forces = np.clip(abs_forces, -1.0, 1.0)

        return label, abs_forces, delta_forces

    def _plot_tactile_clip(self, clip):
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, self.num_frames, figsize=(20, 5))
        for i in range(self.num_frames):
            axs[i].imshow(clip[i].permute(1, 2, 0))
            axs[i].axis("off")
        plt.savefig("tactile_clip.png")

    def _get_tactile_images(self, idx_trajectory, idx_sample):
        t_indexes = self.trajectories[idx_trajectory]["indexes"]
        len_trajectory = len(t_indexes)

        sample_images = []
        for i in self.frames_concat_idx:
            idx_sample = np.clip(idx_sample - i, 0, len_trajectory - 1)
            frame_index = t_indexes[idx_sample]
            image = load_sample_from_buf(self.frames[frame_index], self.bg)
            image = self.transform_resize(image)
            sample_images.append(image)

        if DEBUG:
            self._plot_tactile_clip(sample_images)

        if self.out_format == "single_image":
            output = sample_images[0]
        elif self.out_format == "video":
            output = torch.stack(sample_images, dim=0)
            output = output.permute(1, 0, 2, 3)
        elif self.out_format == "concat_ch_img":
            output = torch.cat(sample_images, dim=1)
        return output

    def _getitem_trajectory(self, idx_trajectory, idx_sample):
        assert idx_trajectory < len(self.trajectories), "Index trajectory out of range"
        assert idx_sample < len(
            self.trajectories[idx_trajectory]["indexes"]
        ), "Index sample out of range"

        inputs = {}
        label, delta_force = self._get_labels(0, idx_trajectory, idx_sample)
        t_indexes = self.trajectories[idx_trajectory]["indexes"]

        inputs = {}
        inputs["image"] = self._get_tactile_images(idx_trajectory, idx_sample)
        inputs["delta_force"] = torch.tensor(delta_force).float()
        inputs["slip_label"] = torch.tensor(label)
        inputs["slip_category_label"] = SLIP_LABELS[label]
        return inputs

    def _get_trajectory_pose_force(self, idx_trajectory):
        assert idx_trajectory < len(self.trajectories), "Index trajectory out of range"
        t_indexes = self.trajectories[idx_trajectory]["indexes"]
        force = self.trajectories[idx_trajectory]["forces"]
        pose = None
        return pose, force
    

def load_forceslip_datasets():
    # load train datasets
    train = []
    for dataset in config.list_datasets:
        train.append(VisionForceSlipDataset(config, dataset))

    full_train_dataset = data.ConcatDataset(train)
    
    # Apply train data budget
    train_data_budget = config.train_data_budget
    if train_data_budget < 1.0:
        train_size = int(len(full_train_dataset) * train_data_budget)
        indices = torch.randperm(len(full_train_dataset))[:train_size]
        train_dataset = data.Subset(full_train_dataset, indices)
    else:
        train_dataset = full_train_dataset

    # load test datasets
    test = []
    for dataset in config.list_datasets_test:
        test.append(VisionForceSlipDataset(config, dataset))

    test_dataset = data.ConcatDataset(test)

    wandb.log({
        "config": OmegaConf.to_yaml(config),
        "train_data_size": len(train_dataset),
        "test_data_size": len(test_dataset),
    })

    return train_dataset, test_dataset
