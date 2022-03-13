# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import torch
import logging


class SchedulerMixin(object):
    def save_checkpoint(
        self,
        ckpt_dir: str = ".",
        weight_name="pytorch_scheduler.bin",
    ):
        state_dict = self.state_dict()
        weight_path = os.path.join(ckpt_dir, weight_name)
        torch.save(state_dict, weight_path)
        logging.info(f"{type(self).__name__} scheduler save checkpoint to {weight_path}")

    def from_checkpoint(
        self,
        ckpt_dir: str = ".",
        weight_name="pytorch_scheduler.bin",
    ):
        weight_path = os.path.join(ckpt_dir, weight_name)
        if not os.path.exists(weight_path):
            return
        state_dict = torch.load(weight_path, map_location="cpu")
        self.load_state_dict(state_dict)
        logging.info(f"{type(self).__name__} scheduler load weight from checkpoint {weight_path}")


import unitorch.cli.scheduler.warmup
