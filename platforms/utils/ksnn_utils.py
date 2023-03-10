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


def run_ksnn(ksnn_model, input):
    input = input[0].transpose(1, 2, 0)
    outputs = ksnn_model.nn_inference([input], platform='ONNX', reorder='2 1 0',
                                      output_format=output_format.OUT_FORMAT_FLOAT32)
    return outputs


def run_head(model, zf, xf):
    zf_pad = F.pad(torch.tensor(zf), (4, 4, 4, 4), "constant", 0)
    inp = torch.cat((zf_pad, torch.tensor(xf)), dim=0)
    inp = inp.permute(1, 2, 0).numpy()

    outputs = model.nn_inference([inp], platform='ONNX', output_tensor=2, reorder='2 1 0',
                                 output_format=output_format.OUT_FORMAT_FLOAT32)
    outputs1 = outputs[0].reshape(1, 2, 16, 16)
    outputs2 = outputs[1].reshape(1, 4, 16, 16)
    return outputs1, outputs2


def run2head(model, zf, xf):
    zf = zf[0].transpose(1, 2, 0)
    xf = xf[0].transpose(1, 2, 0)

    outputs = model.nn_inference([zf, xf], platform='ONNX', input_tensor=2, output_tensor=2, reorder='2 1 0',
                                 output_format=output_format.OUT_FORMAT_FLOAT32)
    outputs1 = outputs[0].reshape(1, 2, 16, 16)
    outputs2 = outputs[1].reshape(1, 4, 16, 16)
    return outputs1, outputs2
