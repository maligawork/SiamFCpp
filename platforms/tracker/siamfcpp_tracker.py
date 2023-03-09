from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from platforms.core.config import cfg
from platforms.tracker.base_tracker import BaseTracker
from siamfcpp.pipeline.utils import (cxywh2xywh, get_crop,
                                     get_subwindow_tracking,
                                     xywh2cxywh, xyxy2cxywh)


class SiamFCppTracker(BaseTracker):
    def __init__(self, model):
        super(SiamFCppTracker, self).__init__()

        self.im_h = None
        self.im_w = None
        self.target_sz = None
        self.target_pos = None
        self.avg_chans = None

        if cfg.windowing == 'cosine':
            window = np.outer(np.hanning(cfg.score_size), np.hanning(cfg.score_size))
            window = window.reshape(-1)
        elif cfg.windowing == 'uniform':
            window = np.ones((cfg.score_size, cfg.score_size))
        else:
            window = np.ones((cfg.score_size, cfg.score_size))
        self.window = window

        self.model = model

    def init(self, img, rect):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """

        self.im_h = img.shape[0]
        self.im_w = img.shape[1]

        box = xywh2cxywh(rect)
        self.target_pos, self.target_sz = box[:2], box[2:]

        self.avg_chans = np.mean(img, axis=(0, 1))

        im_z_crop, _ = get_crop(
            img,
            self.target_pos,
            self.target_sz,
            cfg.z_size,
            avg_chans=self.avg_chans,
            context_amount=cfg.context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )

        im_z_crop = self._to_bchw(im_z_crop)

        self.model.template(im_z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        im_x_crop, scale_x = get_crop(
            img,
            self.target_pos,
            self.target_sz,
            cfg.z_size,
            x_size=cfg.x_size,
            avg_chans=self.avg_chans,
            context_amount=cfg.context_amount,
            func_get_subwindow=get_subwindow_tracking,
        )

        im_x_crop = self._to_bchw(im_x_crop)

        score, box, cls, ctr = self.model.track(im_x_crop)

        box = box[0]
        box_wh = xyxy2cxywh(box)
        score = score[0][:, 0]
        cls = cls[0]
        ctr = ctr[0]

        best_pscore_id, pscore, penalty = self._postprocess_score(
            score, box_wh, self.target_sz, scale_x)

        new_target_pos, new_target_sz = self._postprocess_box(
            best_pscore_id, score, box_wh, self.target_pos, self.target_sz,
            scale_x, cfg.x_size, penalty)

        new_target_pos, new_target_sz = self._restrict_box(
            new_target_pos, new_target_sz)

        self.target_pos, self.target_sz = new_target_pos, new_target_sz

        track_rect = cxywh2xywh(np.concatenate([self.target_pos, self.target_sz],
                                               axis=-1))

        return {'bbox': track_rect, 'best_score': pscore[best_pscore_id]}

    def _postprocess_score(self, score, box_wh, target_sz, scale_x):
        r"""
        Perform SiameseRPN-based tracker's post-processing of score
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_sz: previous state (w & h)
        :param scale_x:
        :return:
            best_pscore_id: index of chosen candidate along axis HW
            pscore: (HW, ), penalized score
            penalty: (HW, ), penalty due to scale/ratio change
        """

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        penalty_k = cfg.penalty_k
        target_sz_in_crop = target_sz * scale_x
        s_c = change(
            sz(box_wh[:, 2], box_wh[:, 3]) /
            (sz_wh(target_sz_in_crop)))  # scale penalty
        r_c = change((target_sz_in_crop[0] / target_sz_in_crop[1]) /
                     (box_wh[:, 2] / box_wh[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1) * penalty_k)
        pscore = penalty * score

        # ipdb.set_trace()
        # cos window (motion model)
        window_influence = cfg.window_influence

        pscore = pscore * (
                1 - window_influence) + self.window * window_influence
        best_pscore_id = np.argmax(pscore)

        return best_pscore_id, pscore, penalty

    @staticmethod
    def _postprocess_box(best_pscore_id, score, box_wh, target_pos,
                         target_sz, scale_x, x_size, penalty):
        r"""
        Perform SiameseRPN-based tracker's post-processing of box
        :param score: (HW, ), score prediction
        :param box_wh: (HW, 4), cxywh, bbox prediction (format changed)
        :param target_pos: (2, ) previous position (x & y)
        :param target_sz: (2, ) previous state (w & h)
        :param scale_x: scale of cropped patch of current frame
        :param x_size: size of cropped patch
        :param penalty: scale/ratio change penalty calculated during score post-processing
        :return:
            new_target_pos: (2, ), new target position
            new_target_sz: (2, ), new target size
        """
        pred_in_crop = box_wh[best_pscore_id, :] / np.float32(scale_x)
        # about np.float32(scale_x)
        # attention!, this casting is done implicitly
        # which can influence final EAO heavily given a model & a set of hyper-parameters

        # box post-postprocessing
        test_lr = cfg.test_lr
        lr = penalty[best_pscore_id] * score[best_pscore_id] * test_lr
        res_x = pred_in_crop[0] + target_pos[0] - (x_size // 2) / scale_x
        res_y = pred_in_crop[1] + target_pos[1] - (x_size // 2) / scale_x
        res_w = target_sz[0] * (1 - lr) + pred_in_crop[2] * lr
        res_h = target_sz[1] * (1 - lr) + pred_in_crop[3] * lr

        new_target_pos = np.array([res_x, res_y])
        new_target_sz = np.array([res_w, res_h])

        return new_target_pos, new_target_sz

    def _restrict_box(self, target_pos, target_sz):
        r"""
        Restrict target position & size
        :param target_pos: (2, ), target position
        :param target_sz: (2, ), target size
        :return:
            target_pos, target_sz
        """
        target_pos[0] = max(0, min(self.im_w, target_pos[0]))
        target_pos[1] = max(0, min(self.im_h, target_pos[1]))
        target_sz[0] = max(cfg.min_w,
                           min(self.im_w, target_sz[0]))
        target_sz[1] = max(cfg.min_h,
                           min(self.im_h, target_sz[1]))

        return target_pos, target_sz

    @staticmethod
    def _to_bchw(im_patch):
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        return im_patch.astype(np.float32)
