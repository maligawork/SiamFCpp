import torch
import numpy as np
from siamfcpp.model.task_head_new.taskhead_impl.track_head import get_box


def get_box_full(cfg, ctr, offsets):
    offsets = np.exp(cfg.si * offsets + cfg.bi) * cfg.total_stride
    offsets = get_box(ctr, torch.Tensor(offsets))
    return offsets.numpy()
