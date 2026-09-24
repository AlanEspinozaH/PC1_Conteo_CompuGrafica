import cv2
import numpy as np

from stabilizer import VideoStabilizer


def textured_frame(height=240, width=320):
    """Crea un fondo determinista con esquinas fáciles de seguir."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(25, height - 20, 35):
        for x in range(25, width - 20, 35):
            color = ((3 * x) % 255, (5 * y) % 255, (x + y) % 255)
            cv2.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), color, -1)
    cv2.putText(
        frame,
        "FONDO",
        (75, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    return frame


def mean_absolute_error(first, second):
    first_f = first.astype(np.float32)
    second_f = second.astype(np.float32)
    return float(np.mean(np.abs(first_f - second_f)))


def test_first_frame_is_not_modified():
    frame = textured_frame()
    stabilizer = VideoStabilizer(border_scale=1.0)

    result = stabilizer.stabilize(frame)

    assert np.array_equal(result, frame)
    assert stabilizer.diagnostics["reason"] == "primer frame"


def test_stabilization_reduces_a_small_translation():
    reference = textured_frame()
    movement = np.float32([[1, 0, 8], [0, 1, -5]])
    moved = cv2.warpAffine(
        reference,
        movement,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    stabilizer = VideoStabilizer(
        smoothing_window=30,
        border_scale=1.0,
        max_translation=20,
        min_inliers=8,
    )

    stabilizer.stabilize(reference)
    corrected = stabilizer.stabilize(moved)

    # Se ignoran bordes porque cualquier warp necesita sintetizar esos píxeles.
    interior = np.s_[20:-20, 20:-20]
    raw_error = mean_absolute_error(reference[interior], moved[interior])
    corrected_error = mean_absolute_error(reference[interior], corrected[interior])

    assert stabilizer.diagnostics["estimated"] is True
    assert stabilizer.diagnostics["inliers"] >= 8
    assert corrected_error < 0.45 * raw_error


def test_blank_frames_use_identity_fallback():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    stabilizer = VideoStabilizer(border_scale=1.0)

    stabilizer.stabilize(frame)
    result = stabilizer.stabilize(frame)

    assert np.array_equal(result, frame)
    assert stabilizer.diagnostics["estimated"] is False
    assert stabilizer.diagnostics["reason"] == "pocos puntos característicos"


def test_exclusion_roi_masks_the_counting_area():
    stabilizer = VideoStabilizer()

    stabilizer.set_exclusion_roi((20, 30, 40, 50), (120, 160, 3), padding=0)

    assert stabilizer.feature_mask is not None
    assert np.all(stabilizer.feature_mask[30:80, 20:60] == 0)
    assert stabilizer.feature_mask[0, 0] == 255


def test_reset_clears_temporal_state():
    stabilizer = VideoStabilizer(border_scale=1.0)
    stabilizer.stabilize(textured_frame())

    stabilizer.reset()

    assert stabilizer.previous_gray is None
    assert stabilizer.feature_mask is None
    assert stabilizer.frame_index == 0
    assert np.array_equal(stabilizer.camera_path, np.eye(3))
