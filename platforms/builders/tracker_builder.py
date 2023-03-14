import importlib
from platforms.core.config import cfg

PLATFORMS = ('onnx', 'onnx_one',  'onnx_head', 'opencv', 'ksnn', 'ksnn_head')


def get_tracker(platform: str):
    if platform not in PLATFORMS:
        raise Exception('Platform doesn\'t exist!')

    cfg.merge_from_file(f'platforms/config/config_{platform}.yaml')
    builder = importlib.import_module(f'platforms.builders.{platform}_builder')

    tracker = builder.create_tracker()
    return tracker
