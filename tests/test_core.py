import cv2
import numpy as np
import pytest

from extract_patches.core import extract_patches, extract_patches_Ms
from extract_patches.laf import Ell2LAF, convert_LAFs, convertLAFs_to_A23format


def test_extract_cv2_keypoint_returns_centered_patch():
    image = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    keypoint = cv2.KeyPoint(32, 32, 8, 0)
    patches = extract_patches([keypoint], image, PS=16, mag_factor=1.0)
    assert len(patches) == 1
    assert patches[0].shape == (16, 16)
    assert patches[0].dtype == image.dtype


def test_unoriented_cv2_keypoint_is_supported():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    patch = extract_patches([cv2.KeyPoint(16, 16, 8)], image, PS=8, mag_factor=1.0)[0]
    assert patch.shape == (8, 8, 3)


def test_small_laf_scale_does_not_select_last_pyramid_level():
    laf = np.array([[[0.1, 0.0, 16.0], [0.0, 0.1, 16.0]]], dtype=np.float32)
    matrices, levels = convert_LAFs(laf, PS=32, mag_factor=1.0)
    assert levels == [0]
    patch = extract_patches_Ms(matrices, np.zeros((64, 64), dtype=np.uint8), levels, 32)[0]
    assert patch.shape == (32, 32)


def test_extract_patches_ms_default_levels_are_not_shared_or_empty():
    matrix = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    patches = extract_patches_Ms([matrix], np.zeros((16, 16), dtype=np.uint8), PS=8)
    assert len(patches) == 1


def test_bad_inputs_raise_value_error():
    with pytest.raises(ValueError, match="unknown input format"):
        extract_patches([], np.zeros((8, 8)), input_format="bad")
    with pytest.raises(ValueError, match="positive definite"):
        Ell2LAF([0, 0, 1, 2, 1])
    with pytest.raises(ValueError, match="LAFs must"):
        convertLAFs_to_A23format(np.zeros((2, 5)))


def test_flat_laf_formats_are_converted():
    flat = np.array([[10, 20, 2, 1, 0, 0, 1]], dtype=float)
    converted = convertLAFs_to_A23format(flat)
    np.testing.assert_allclose(converted[0], [[2, 0, 10], [0, 2, 20]])

