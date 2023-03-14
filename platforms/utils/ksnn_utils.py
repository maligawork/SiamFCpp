import os
import torch
import torch.nn.functional as F
from ksnn.api import KSNN
from ksnn.types import *


def load_ksnn(main_folder, model_path, level=0):
    model = KSNN('VIM3')
    model.nn_init(library=os.path.join(main_folder, model_path, 'libnn_' + model_path + '.so'),
                  model=os.path.join(main_folder, model_path, model_path + '.nb'),
                  level=level)
    return model


def run_ksnn(ksnn_model, input, output_tensor=1):
    input = input[0].transpose(1, 2, 0)
    outputs = ksnn_model.nn_inference([input], platform='ONNX', output_tensor=output_tensor, reorder='2 1 0',
                                      output_format=output_format.OUT_FORMAT_FLOAT32)
    return outputs


def run_head(model, inp1, inp2):
    inp1 = inp1[0].permute(1, 2, 0).numpy()
    inp2 = inp2[0].permute(1, 2, 0).numpy()

    outputs = model.nn_inference([inp1, inp2], platform='ONNX', output_tensor=4, reorder='2 1 0',
                                 output_format=output_format.OUT_FORMAT_FLOAT32)

    csl_score = outputs[0].reshape(1, 289, 1)
    ctr_score = outputs[1].reshape(1, 289, 1)
    offsets   = outputs[2].reshape(1, 4, 17, 17)
    fea       = outputs[3].reshape(1, 256, 17, 17)

    return csl_score, ctr_score, offsets, fea
