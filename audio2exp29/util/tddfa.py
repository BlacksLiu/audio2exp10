# coding: utf-8

import numpy as np
import cv2
import os.path as osp
import pickle
__author__ = 'cleardusk'

import sys

sys.path.append('..')


def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def _parse_param(param):
    """matrix pose form
    param: shape=(trans_dim+shape_dim+exp_dim,), i.e., 62 = 12 + 40 + 10
    """

    # pre-defined templates for parameter
    n = param.shape[0]
    if n == 62:
        trans_dim, shape_dim, exp_dim = 12, 40, 10
    elif n == 72:
        trans_dim, shape_dim, exp_dim = 12, 40, 20
    elif n == 141:
        trans_dim, shape_dim, exp_dim = 12, 100, 29
    else:
        raise Exception(f'Undefined templated param parsing rule')

    R_ = param[:trans_dim].reshape(3, -1)
    R = R_[:, :3]
    offset = R_[:, -1].reshape(3, 1)
    alpha_shp = param[trans_dim:trans_dim + shape_dim].reshape(-1, 1)
    alpha_exp = param[trans_dim + shape_dim:].reshape(-1, 1)

    return R, offset, alpha_shp, alpha_exp


class BFMModel(object):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            # self.tri = _load(make_abs_path('../configs/tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
            self.tri = _load(osp.join(osp.dirname(bfm_fp), 'tri.pkl'))
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

    def recon_vers(self, param, roi_box, size=120):
        R, offset, alpha_shp, alpha_exp = _parse_param(param)
        pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
            reshape(3, -1, order='F') + offset
        pts3d = similar_transform(pts3d, roi_box, size)
        return pts3d


def similar_transform(pts3d, roi_box, size):
    pts3d[0, :] -= 1  # for Python compatibility
    pts3d[2, :] -= 1
    pts3d[1, :] = size - pts3d[1, :]

    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / size
    scale_y = (ey - sy) / size
    pts3d[0, :] = pts3d[0, :] * scale_x + sx
    pts3d[1, :] = pts3d[1, :] * scale_y + sy
    s = (scale_x + scale_y) / 2
    pts3d[2, :] *= s
    pts3d[2, :] -= np.min(pts3d[2, :])
    return np.array(pts3d, dtype=np.float32)


def draw_landmarks(img, pts3d, color=(255, 255, 255), thickness=2):
    """Draw landmarks using cv2"""
    img = img.copy()
    pts3d = np.transpose(pts3d, axes=[1, 0])[:, :2]  # (3, 68) -> (68, 3)
    pts3d = pts3d.astype(np.int32)
    cv2.polylines(img, [pts3d[0: 17].reshape(-1, 1, 2)], False, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[17: 22].reshape(-1, 1, 2)], False, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[22: 27].reshape(-1, 1, 2)], False, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[27: 31].reshape(-1, 1, 2)], False, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[31: 36].reshape(-1, 1, 2)], True, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[36: 42].reshape(-1, 1, 2)], True, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[42: 48].reshape(-1, 1, 2)], True, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[48: 60].reshape(-1, 1, 2)], True, color=color, thickness=thickness)
    cv2.polylines(img, [pts3d[60: 68].reshape(-1, 1, 2)], True, color=color, thickness=thickness)
    return img


mean_std_param = _load('data/bfm/param_mean_std_62d_120x120.pkl')
mean_param = mean_std_param['mean']
std_param = mean_std_param['std']
print(mean_param, std_param)
