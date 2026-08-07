"""Local affine frame conversion helpers."""

import math
from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np


def _pyramid_level(scale: float) -> int:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("affine scale must be finite and greater than zero")
    return max(0, int(math.log(scale, 2)))


def convert_LAFs(kps: List, PS: int, mag_factor: float) -> Tuple[List[np.ndarray], List[int]]:
    Ms, pyr_idxs = [], []
    for kp in kps:
        x, y = kp[0, 2], kp[1, 2]
        Ai = 2.0 * mag_factor * kp[:2, :2] / PS
        pyr_idx = _pyramid_level(math.sqrt(abs(np.linalg.det(Ai))))
        d_factor = float(2 ** pyr_idx)
        Ai /= d_factor
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = Ai
        M[0, 2] = (-Ai[0, 0] - Ai[0, 1]) * PS / 2 + x / d_factor
        M[1, 2] = (-Ai[1, 0] - Ai[1, 1]) * PS / 2 + y / d_factor
        Ms.append(M)
        pyr_idxs.append(pyr_idx)
    return Ms, pyr_idxs


def Ell2LAF(ell: Union[List, np.ndarray]) -> np.ndarray:
    ell = np.asarray(ell, dtype=float)
    if ell.shape != (5,):
        raise ValueError("ellipse must contain [x, y, a, b, c]")
    a, b, c = ell[2:]
    determinant = a * c - b * b
    if a <= 0 or c <= 0 or determinant <= 0:
        raise ValueError("ellipse matrix must be positive definite")
    A23 = np.zeros((2, 3))
    A23[:, 2] = ell[:2]
    scale = np.sqrt(np.sqrt(determinant))
    ia, ib, ic = invSqrt(a, b, c)
    A = np.array([[ia, ib], [ib, ic]]) / scale
    scale = np.sqrt(np.linalg.det(A))
    A23[:2, :2] = rectifyAffineTransformationUpIsUp(A / scale) * scale
    return A23


def invSqrt(a, b, c):
    values, vectors = np.linalg.eigh(np.array([[a, b], [b, c]], dtype=float))
    if np.any(values <= 0):
        raise ValueError("matrix must be positive definite")
    result = vectors @ np.diag(1.0 / np.sqrt(values)) @ vectors.T
    determinant_scale = np.sqrt(result[0, 0] * result[1, 1] - result[0, 1] ** 2)
    result /= determinant_scale
    return result[0, 0], result[0, 1], result[1, 1]


def rectifyAffineTransformationUpIsUp(A: np.ndarray) -> np.ndarray:
    determinant = abs(np.linalg.det(A))
    first_row_norm = np.linalg.norm(A[0])
    if determinant <= 0 or first_row_norm <= 0:
        raise ValueError("affine matrix must be non-singular")
    det = np.sqrt(determinant)
    return np.array([
        [first_row_norm / det, 0.0],
        [(A[1, 1]*A[0, 1] + A[1, 0]*A[0, 0]) / (first_row_norm*det), det/first_row_norm],
    ])


def convert_ellipse_keypoints(ells: Union[List, np.ndarray], PS: int, mag_factor: float):
    Ms, pyr_idxs = [], []
    for ell in ells:
        LAF = Ell2LAF(ell)
        x, y = LAF[0, 2], LAF[1, 2]
        Ai = mag_factor * LAF[:2, :2] / PS
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


def ells2LAFs(ells: Union[List, np.ndarray]) -> np.ndarray:
    return np.stack([Ell2LAF(ell) for ell in ells]) if len(ells) else np.empty((0, 2, 3))


def LAF2pts(LAF: np.ndarray, n_pts: int = 50) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, n_pts)
    points = np.vstack(([0] + list(np.sin(angles)), [0] + list(np.cos(angles)), np.ones(n_pts + 1)))
    homogeneous = np.vstack((LAF, [0, 0, 1])) @ points
    return (homogeneous[:2] / homogeneous[2]).T


def convertLAFs_to_A23format(LAFs: np.ndarray) -> np.ndarray:
    LAFs = np.asarray(LAFs)
    if LAFs.ndim == 3 and LAFs.shape[1:] == (2, 3):
        return deepcopy(LAFs)
    if LAFs.ndim != 2 or LAFs.shape[1] not in (6, 7):
        raise ValueError("LAFs must have shape (N,2,3), (N,6), or (N,7)")
    work = np.zeros((LAFs.shape[0], 2, 3))
    work[:, :, 2] = LAFs[:, :2]
    offset = 3 if LAFs.shape[1] == 7 else 2
    work[:, :, :2] = LAFs[:, offset:].reshape(-1, 2, 2)
    if LAFs.shape[1] == 7:
        work[:, :, :2] *= LAFs[:, 2, None, None]
    return work


def LAFs2ell(in_LAFs):
    LAFs = convertLAFs_to_A23format(in_LAFs)
    ellipses = np.zeros((len(LAFs), 5))
    for i, LAF in enumerate(LAFs):
        linear = LAF[:2, :2]
        scale = np.sqrt(np.linalg.det(linear))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("LAF matrices must have positive determinants")
        u, singular_values, _ = np.linalg.svd(linear / scale, full_matrices=True)
        weights = 1.0 / (singular_values ** 2 * scale ** 2)
        A = u @ np.diag(weights) @ u.T
        ellipses[i] = [LAF[0, 2], LAF[1, 2], A[0, 0], A[0, 1], A[1, 1]]
    return ellipses


def visualize_LAFs(img: np.ndarray, LAFs: np.ndarray, color: str = "r") -> None:
    work_LAFs = convertLAFs_to_A23format(LAFs)
    plt.figure()
    plt.imshow(img)
    for LAF in work_LAFs:
        ell = LAF2pts(LAF)
        plt.plot(ell[:, 0], ell[:, 1], color)
    plt.show()

