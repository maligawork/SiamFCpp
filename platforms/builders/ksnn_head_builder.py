from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np

from platforms.core.config import cfg
from platforms.utils.ksnn_utils import load_ksnn, run_ksnn
from platforms.utils.opencv_utils import load_opencv, run_opencv
from platforms.utils.onnx_utils import load_onnx, run_onnx
from platforms.tracker.tracker_builder import build_tracker
from siamfcpp.model.task_head_new.taskhead_impl.track_head import get_xy_ctr
from siamfcpp.utils.box_utils import get_box_full

import cv2
from siamfcpp.model.common_opr.common_block import xcorr_depthwise


class ModelBuilder:
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.c_z_k = None
        self.r_z_k = None
        self.ksnn_models_path = cfg.KSNN_MODELS_PATH
        self.backbone_init_folder = cfg.KSNN_BACKBONE_INIT
        self.backbone_folder = cfg.KSNN_BACKBONE
        self.head_path = cfg.ONNX_HEAD

        self.backend = cv2.dnn.DNN_BACKEND_TIMVX
        # self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
        self.target = cv2.dnn.DNN_TARGET_NPU
        # self.target = cv2.dnn.DNN_TARGET_CPU

        if cfg.CUDA:
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'

        self.backbone_init = load_ksnn(self.ksnn_models_path, self.backbone_init_folder)
        self.backbone = load_ksnn(self.ksnn_models_path, self.backbone_folder)
        self.ban_head = load_opencv(self.head_path, self.backend, self.target)
        # self.ban_head = load_onnx(self.head_path, provider)
        self.ctr = get_xy_ctr(cfg.score_size, cfg.score_offset, cfg.total_stride)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def template(self, z):
        c_z_k, r_z_k = run_ksnn(self.backbone_init, z, output_tensor=2)
        c_z_k = c_z_k.reshape(1, 256, 4, 4).astype(np.float32)
        r_z_k = r_z_k.reshape(1, 256, 4, 4).astype(np.float32)
        self.c_z_k = c_z_k
        self.r_z_k = r_z_k

    def track(self, x):
        c_x, r_x = run_ksnn(self.backbone, x, output_tensor=2)
        c_x = c_x.reshape(1, 256, 26, 26).astype(np.float32)
        r_x = r_x.reshape(1, 256, 26, 26).astype(np.float32)

        c_out = xcorr_depthwise(torch.Tensor(c_x), torch.Tensor(self.c_z_k))
        r_out = xcorr_depthwise(torch.Tensor(r_x), torch.Tensor(self.r_z_k))

        out = torch.cat([c_out, r_out], dim=1)

        fcos_cls_score_final, fcos_ctr_score_final, offsets, corr_fea = run_opencv(self.ban_head, out.numpy(),
                                                                                   ['csl_score', 'ctr_score',
                                                                                    'offsets', 'fea'])

        # fcos_cls_score_final, fcos_ctr_score_final, offsets, corr_fea = run_onnx(self.ban_head,
        #                                                                          {'input': out.numpy()})

        fcos_bbox_final = get_box_full(cfg, self.ctr, offsets)

        fcos_cls_prob_final = self.sigmoid(fcos_cls_score_final)
        fcos_ctr_prob_final = self.sigmoid(fcos_ctr_score_final)
        fcos_score_final = fcos_cls_prob_final * fcos_ctr_prob_final

        return fcos_score_final, fcos_bbox_final, fcos_cls_prob_final, fcos_ctr_prob_final


def create_tracker():
    model = ModelBuilder()
    return build_tracker(model)
