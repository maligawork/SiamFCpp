import os
import time
import argparse

import cv2
import imageio

import torch
from siamfcpp.config.config import specify_task
from siamfcpp.config.config import cfg
from siamfcpp.model import builder as model_builder
from siamfcpp.pipeline import builder as pipeline_builder
from siamfcpp.utils import complete_path_wt_root_in_cfg


def create_tracker():
    dev = torch.device("cuda:0")

    config = 'models/siamfcpp/test/vot/siamfcpp_alexnet.yaml'
    model_path = 'models/snapshots/siamfcpp_alexnet-got/epoch-17.pkl'

    exp_cfg_path = os.path.realpath(config)
    cfg.merge_from_file(exp_cfg_path)
    cfg.test.track.model.task_model.SiamTrack.pretrain_model_path = model_path

    ROOT_PATH = os.getcwd()
    root_cfg = complete_path_wt_root_in_cfg(cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    _, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    model = model_builder.build('track', task_cfg.model)
    tracker = pipeline_builder.build('track', task_cfg.pipeline, model)
    tracker.set_device(dev)

    return tracker


def main(input_video: str, output_video: str, init_box: str, platform: str):

    tracker = create_tracker()

    cap = cv2.VideoCapture(input_video)
    success, _ = cap.read()
    if not success:
        print('Read frame from {} failed.'.format(input_video))

    writer = imageio.get_writer(output_video, fps=30)

    time_counter = 0
    idx = 0
    while True:
        ret, img = cap.read()

        if idx > 300:
            break

        if img is None:
            break

        if idx == 0:
            init_box = [int(p) for p in init_box.split()]
            tracker.init(img, init_box)
        else:
            start_time = time.time()
            outputs = tracker.update(img)
            time_counter += time.time() - start_time
            pred_bbox = outputs

            bbox = list(map(int, pred_bbox))
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 255), 3)
            cv2.putText(img, f'{idx}', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        print(idx)
        idx += 1

    print('Fps: {}'.format(round(idx / time_counter, 3)))
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the tracker on your video.')
    # parser.add_argument('--platform', type=str, help='platform')
    parser.add_argument('--input_video', type=str, default='videos/kv.mp4', help='path to a input video file.')
    parser.add_argument('--output_video', type=str, default='videos/out.mp4', help='path to a output video file.')
    parser.add_argument('--init_box', type=str, default='674 426 219 130', help='initial box coordinates xywh')

    args = parser.parse_args()

    main(args.input_video, args.output_video, args.init_box, '')
