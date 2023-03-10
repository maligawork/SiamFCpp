from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np

from platforms.core.config import cfg
from platforms.utils.opencv_utils import load_opencv, run_opencv
from platforms.utils.onnx_utils import load_onnx, run_onnx
from platforms.tracker.tracker_builder import build_tracker

from siamfcpp.model.common_opr.common_block import xcorr_depthwise


class ModelBuilder:
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.c_z_k = None
        self.r_z_k = None
        self.backbone_init_path = cfg.OPENCV_BACKBONE_INIT
        self.backbone_path = cfg.OPENCV_BACKBONE
        self.head_path = cfg.OPENCV_HEAD

        # self.backend = cv2.dnn.DNN_BACKEND_TIMVX
        self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        # self.target = cv2.dnn.DNN_TARGET_NPU
        self.target = cv2.dnn.DNN_TARGET_CPU

        if cfg.CUDA:
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'

        self.backbone_init = load_opencv(self.backbone_init_path, self.backend, self.target)
        self.backbone = load_opencv(self.backbone_path, self.backend, self.target)
        self.ban_head = load_onnx(self.head_path, provider)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def template(self, z):
        c_z_k, r_z_k = run_opencv(self.backbone_init, z, ['c_z_k', 'r_z_k'])
        self.c_z_k = c_z_k
        self.r_z_k = r_z_k

    def track(self, x):
        c_x, r_x = run_opencv(self.backbone, x, ['c_x', 'r_x'])

        c_out = xcorr_depthwise(torch.Tensor(c_x), torch.Tensor(self.c_z_k))
        r_out = xcorr_depthwise(torch.Tensor(r_x), torch.Tensor(self.r_z_k))

        fcos_cls_score_final, fcos_ctr_score_final, fcos_bbox_final, corr_fea = run_onnx(self.ban_head,
                                                                                         {'input1': c_out.numpy(),
                                                                                          'input2': r_out.numpy()})

        fcos_cls_prob_final = self.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = self.sigmoid(fcos_ctr_score_final)
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

        return fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final


def create_tracker():
    model = ModelBuilder()
    return build_tracker(model)
