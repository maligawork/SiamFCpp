# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = ""

__C.CUDA = False

__C.PYTORCH_MODEL = ''

__C.ONNX_BACKBONE_INIT = ''
__C.ONNX_BACKBONE = ''
__C.ONNX_HEAD = ''

__C.OPENCV_BACKBONE_INIT = ''
__C.OPENCV_BACKBONE = ''
__C.OPENCV_HEAD = ''

__C.KSNN_MODELS_PATH = ''
__C.KSNN_BACKBONE_INIT = ''
__C.KSNN_BACKBONE = ''
__C.KSNN_HEAD = ''

__C.TANGENT_BACKBONE_INIT = ''
__C.TANGENT_BACKBONE = ''

__C.TRACK = CN()
__C.TRACK.TYPE = 'SiamFCppTracker'

__C.z_size = 127
__C.x_size = 303
__C.context_amount = 0.5
__C.windowing = 'cosine'
__C.score_size = 17
__C.penalty_k = 0.04
__C.window_influence = 0.21
__C.test_lr = 0.52
__C.min_w = 10
__C.min_h = 10
