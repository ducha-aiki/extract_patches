"""Patch extraction routines."""

import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .laf import convert_LAFs, convert_ellipse_keypoints


def _pyramid_level(scale: float) -> int:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("keypoint scale must be finite and greater than zero")
    return max(0, int(math.log(scale, 2)))


def extract_patches(
    kpts: Union[List, np.ndarray],
    img: np.ndarray,
    PS: int = 32,
    mag_factor: float = 10.0,
    input_format: str = "cv2",
) -> List[np.ndarray]:
    """Extract square patches for keypoints in a supported input format."""
    if PS <= 0:
        raise ValueError("PS must be greater than zero")
    if img is None or not isinstance(img, np.ndarray) or img.ndim not in (2, 3):
        raise ValueError("img must be a two- or three-dimensional NumPy array")

    if input_format == "cv2":
        Ms, pyr_idxs = convert_cv2_keypoints(kpts, PS, mag_factor)
    elif input_format == "cv2+A":
        Ms, pyr_idxs = convert_cv2_plus_A_keypoints(kpts[0], kpts[1], PS, mag_factor)
    elif input_format in ("ellipse", "xyabc"):
        _require_shape(kpts, 5, input_format)
        Ms, pyr_idxs = convert_ellipse_keypoints(kpts, PS, mag_factor)
    elif input_format == "xyA":
        _require_shape(kpts, 6, input_format)
        Ms, pyr_idxs = convert_xyA(kpts, PS, mag_factor)
    elif input_format == "LAF":
        kpts = np.asarray(kpts)
        if kpts.ndim == 4:
            if kpts.shape[0] == 1:
                kpts = kpts.squeeze(0)
            elif kpts.shape[1] == 1:
                kpts = kpts.squeeze(1)
            else:
                raise ValueError("LAF array must have a singleton batch dimension")
        if kpts.ndim != 3 or kpts.shape[1:] != (2, 3):
            raise ValueError("LAF array must have shape (N, 2, 3)")
        Ms, pyr_idxs = convert_LAFs(kpts, PS, mag_factor)
    else:
        raise ValueError("unknown input format: {}".format(input_format))
    return extract_patches_Ms(Ms, img, pyr_idxs, PS)


def _require_shape(kpts: np.ndarray, columns: int, input_format: str) -> None:
    if not isinstance(kpts, np.ndarray) or kpts.ndim != 2 or kpts.shape[1] != columns:
        raise ValueError("{} keypoints must have shape (N, {})".format(input_format, columns))


def build_image_pyramid(img: np.ndarray, min_size: int) -> List[np.ndarray]:
    img_pyr = [img]
    cur_img = img
    while min(cur_img.shape[:2]) > min_size:
        cur_img = cv2.pyrDown(cur_img)
        img_pyr.append(cur_img)
    return img_pyr


def extract_patches_Ms(
    Ms: Sequence[np.ndarray],
    img: np.ndarray,
    pyr_idxs: Optional[Sequence[int]] = None,
    PS: int = 32,
) -> List[np.ndarray]:
    if pyr_idxs is None:
        pyr_idxs = [0] * len(Ms)
    if len(Ms) != len(pyr_idxs):
        raise ValueError("Ms and pyr_idxs must have the same length")
    img_pyr = build_image_pyramid(img, max(1, PS // 2))
    max_pyr_idx = len(img_pyr) - 1
    patches = []
    for M, pyr_idx in zip(Ms, pyr_idxs):
        level = min(max_pyr_idx, max(0, int(pyr_idx)))
        patch = cv2.warpAffine(
            img_pyr[level], np.asarray(M, dtype=np.float32), (PS, PS),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS,
            borderMode=cv2.BORDER_REPLICATE,
        )
        patches.append(patch)
    return patches


def convert_cv2_keypoints(kps: List, PS: int, mag_factor: float) -> Tuple[List[np.ndarray], List[int]]:
    Ms, pyr_idxs = [], []
    for kp in kps:
        x, y = kp.pt
        scale = mag_factor * kp.size / PS
        pyr_idx = _pyramid_level(scale)
        d_factor = float(2 ** pyr_idx)
        scale /= d_factor
        angle = kp.angle if kp.angle >= 0 else 0.0
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        M = np.array([
            [scale * cos, -scale * sin, (-scale * cos + scale * sin) * PS / 2 + x / d_factor],
            [scale * sin, scale * cos, (-scale * sin - scale * cos) * PS / 2 + y / d_factor],
        ], dtype=np.float32)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def convert_cv2_plus_A_keypoints(kps: List, A: np.ndarray, PS: int, mag_factor: float):
    A = np.asarray(A)
    if A.shape != (len(kps), 2, 2):
        raise ValueError("A must have shape (len(kps), 2, 2)")
    Ms, pyr_idxs = [], []
    for kp, affine in zip(kps, A):
        x, y = kp.pt
        scale = mag_factor * kp.size / PS
        pyr_idx = _pyramid_level(scale)
        d_factor = float(2 ** pyr_idx)
        scale /= d_factor
        angle = kp.angle if kp.angle >= 0 else 0.0
        cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        Ai = np.matmul(np.array([[scale*cos, -scale*sin], [scale*sin, scale*cos]]), affine)
        M = np.column_stack((Ai, [
            (-Ai[0, 0] - Ai[0, 1]) * PS / 2 + x / d_factor,
            (-Ai[1, 0] - Ai[1, 1]) * PS / 2 + y / d_factor,
        ])).astype(np.float32)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def convert_xyA(kps: np.ndarray, PS: int, mag_factor: float):
    Ms, pyr_idxs = [], []
    for kp in kps:
        x, y = kp[:2]
        Ai = mag_factor * kp[2:].reshape(2, 2) / PS
        pyr_idx = _pyramid_level(math.sqrt(abs(np.linalg.det(Ai))))
        d_factor = float(2 ** pyr_idx)
        Ai /= d_factor
        M = np.column_stack((Ai, [
            (-Ai[0, 0] - Ai[0, 1]) * PS / 2 + x / d_factor,
            (-Ai[1, 0] - Ai[1, 1]) * PS / 2 + y / d_factor,
        ])).astype(np.float32)
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs

