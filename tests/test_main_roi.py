import cv2
import numpy as np

from main import load_roi, save_roi, select_scaled_roi


def test_roi_json_round_trip(tmp_path):
    roi_path = tmp_path / "roi.json"

    save_roi(roi_path, (10, 20, 300, 400))

    assert load_roi(roi_path) == (10, 20, 300, 400)


def test_vertical_preview_coordinates_return_to_original_scale(monkeypatch):
    vertical_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

    monkeypatch.setattr(
        cv2,
        "selectROI",
        lambda *args, **kwargs: (10, 20, 30, 40),
    )

    # La altura se reduce de 1920 a 800: escala = 5/12.
    roi = select_scaled_roi(vertical_frame, max_width=1200, max_height=800)

    assert roi == (24, 48, 72, 96)
