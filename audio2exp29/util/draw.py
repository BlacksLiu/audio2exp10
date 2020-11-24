import numpy as np
import cv2


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

