"""Estabilización causal de video mediante movimiento global del fondo.

El estabilizador estima una transformación rígida entre fotogramas crudos,
acumula la trayectoria de la cámara y la suaviza con una media exponencial.
Después lleva el fotograma actual desde la trayectoria observada hacia la
trayectoria suavizada.

Está pensado para vibraciones leves de una cámara casi fija. Un paneo amplio
introduce regiones que no existían en fotogramas anteriores y no puede
corregirse completamente con este método.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import cv2
import numpy as np


class VideoStabilizer:
    """Estabilizador en línea basado en Shi–Tomasi, LK y RANSAC."""

    def __init__(
        self,
        smoothing_window: int = 30,
        max_translation: float = 50.0,
        max_rotation_degrees: float = 5.0,
        border_scale: float = 1.02,
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 20,
        min_inliers: int = 10,
        forward_backward_threshold: float = 1.5,
    ) -> None:
        if smoothing_window < 1:
            raise ValueError("smoothing_window debe ser mayor o igual que 1")
        if not 1.0 <= border_scale <= 1.20:
            raise ValueError("border_scale debe estar entre 1.0 y 1.20")

        self.alpha = 2.0 / (smoothing_window + 1.0)
        self.max_translation = float(max_translation)
        self.max_rotation_radians = math.radians(max_rotation_degrees)
        self.border_scale = float(border_scale)
        self.max_corners = int(max_corners)
        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.min_inliers = int(min_inliers)
        self.forward_backward_threshold = float(forward_backward_threshold)

        self.previous_gray: Optional[np.ndarray] = None
        self.feature_mask: Optional[np.ndarray] = None

        # Matriz que lleva coordenadas del primer frame al frame crudo actual.
        self.camera_path = np.eye(3, dtype=np.float64)
        # Trayectoria deseada, suavizada: tx, ty y ángulo acumulado.
        self.smoothed_path = np.zeros(3, dtype=np.float64)

        self.frame_index = 0
        self.diagnostics = self._empty_diagnostics("sin inicializar")

    @staticmethod
    def _empty_diagnostics(reason: str) -> dict:
        return {
            "estimated": False,
            "reason": reason,
            "tracked_points": 0,
            "inliers": 0,
            "dx": 0.0,
            "dy": 0.0,
            "angle_degrees": 0.0,
            "correction_x": 0.0,
            "correction_y": 0.0,
        }

    def reset(self) -> None:
        """Elimina todo el estado temporal para comenzar otro video."""
        self.previous_gray = None
        self.feature_mask = None
        self.camera_path = np.eye(3, dtype=np.float64)
        self.smoothed_path = np.zeros(3, dtype=np.float64)
        self.frame_index = 0
        self.diagnostics = self._empty_diagnostics("reiniciado")

    def set_exclusion_roi(
        self,
        roi: Sequence[int],
        frame_shape: Sequence[int],
        padding: int = 5,
    ) -> None:
        """Evita usar puntos móviles de la ROI para estimar la cámara.

        La máscara conserva como candidatos los píxeles fuera de la ROI. Si la
        ROI ocupa casi toda la imagen, es preferible no llamar este método.
        """
        if len(roi) != 4:
            raise ValueError("roi debe contener x, y, ancho y alto")

        height, width = int(frame_shape[0]), int(frame_shape[1])
        x, y, w, h = map(int, roi)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)

        mask = np.full((height, width), 255, dtype=np.uint8)
        mask[y1:y2, x1:x2] = 0

        # Evita activar una máscara que deje menos del 15 % del frame.
        if cv2.countNonZero(mask) >= 0.15 * width * height:
            self.feature_mask = mask

    @staticmethod
    def _to_homogeneous(affine: np.ndarray) -> np.ndarray:
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = affine
        return matrix

    @staticmethod
    def _rigid_matrix(dx: float, dy: float, angle: float) -> np.ndarray:
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array(
            [
                [cos_a, -sin_a, dx],
                [sin_a, cos_a, dy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _path_parameters(matrix: np.ndarray) -> np.ndarray:
        return np.array(
            [
                matrix[0, 2],
                matrix[1, 2],
                math.atan2(matrix[1, 0], matrix[0, 0]),
            ],
            dtype=np.float64,
        )

    def _estimate_motion(self, current_gray: np.ndarray) -> Optional[np.ndarray]:
        previous_points = cv2.goodFeaturesToTrack(
            self.previous_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=3,
            mask=self.feature_mask,
        )
        if previous_points is None or len(previous_points) < self.min_inliers:
            self.diagnostics = self._empty_diagnostics("pocos puntos característicos")
            return None

        lk_parameters = {
            "winSize": (21, 21),
            "maxLevel": 3,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        }
        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            current_gray,
            previous_points,
            None,
            **lk_parameters,
        )
        if current_points is None or forward_status is None:
            self.diagnostics = self._empty_diagnostics("falló el flujo óptico")
            return None

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self.previous_gray,
            current_points,
            None,
            **lk_parameters,
        )
        if backward_points is None or backward_status is None:
            self.diagnostics = self._empty_diagnostics("falló la validación inversa")
            return None

        forward_ok = forward_status.reshape(-1) == 1
        backward_ok = backward_status.reshape(-1) == 1
        backward_error = np.linalg.norm(
            previous_points.reshape(-1, 2) - backward_points.reshape(-1, 2),
            axis=1,
        )
        reliable = (
            forward_ok
            & backward_ok
            & np.isfinite(backward_error)
            & (backward_error <= self.forward_backward_threshold)
        )

        good_previous = previous_points.reshape(-1, 2)[reliable]
        good_current = current_points.reshape(-1, 2)[reliable]
        tracked_points = len(good_previous)
        if tracked_points < self.min_inliers:
            self.diagnostics = self._empty_diagnostics("pocos puntos confiables")
            self.diagnostics["tracked_points"] = tracked_points
            return None

        transformation, inlier_mask = cv2.estimateAffinePartial2D(
            good_previous,
            good_current,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
        if transformation is None or inlier_mask is None:
            self.diagnostics = self._empty_diagnostics("RANSAC no estimó transformación")
            self.diagnostics["tracked_points"] = tracked_points
            return None

        inliers = int(inlier_mask.sum())
        if inliers < self.min_inliers:
            self.diagnostics = self._empty_diagnostics("pocos inliers de RANSAC")
            self.diagnostics.update({"tracked_points": tracked_points, "inliers": inliers})
            return None

        dx = float(transformation[0, 2])
        dy = float(transformation[1, 2])
        angle = math.atan2(transformation[1, 0], transformation[0, 0])

        if math.hypot(dx, dy) > self.max_translation:
            self.diagnostics = self._empty_diagnostics("traslación implausible")
            self.diagnostics.update({"tracked_points": tracked_points, "inliers": inliers})
            return None
        if abs(angle) > self.max_rotation_radians:
            self.diagnostics = self._empty_diagnostics("rotación implausible")
            self.diagnostics.update({"tracked_points": tracked_points, "inliers": inliers})
            return None

        self.diagnostics = {
            "estimated": True,
            "reason": "ok",
            "tracked_points": tracked_points,
            "inliers": inliers,
            "dx": dx,
            "dy": dy,
            "angle_degrees": math.degrees(angle),
            "correction_x": 0.0,
            "correction_y": 0.0,
        }
        # Se elimina la escala estimada para evitar zoom acumulativo.
        return self._rigid_matrix(dx, dy, angle)

    def _zoom_matrix(self, width: int, height: int) -> np.ndarray:
        if self.border_scale == 1.0:
            return np.eye(3, dtype=np.float64)
        zoom = cv2.getRotationMatrix2D(
            (width / 2.0, height / 2.0),
            0.0,
            self.border_scale,
        )
        return self._to_homogeneous(zoom)

    def stabilize(self, frame: np.ndarray) -> np.ndarray:
        """Devuelve una copia estabilizada del frame actual."""
        if frame is None or frame.size == 0:
            raise ValueError("frame no puede estar vacío")

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]

        if self.previous_gray is None:
            self.previous_gray = current_gray
            self.frame_index = 1
            self.diagnostics = self._empty_diagnostics("primer frame")
            return frame.copy()

        raw_motion = self._estimate_motion(current_gray)
        if raw_motion is None:
            raw_motion = np.eye(3, dtype=np.float64)

        # P_t = T_(t-1 -> t) P_(t-1): trayectoria observada desde el inicio.
        self.camera_path = raw_motion @ self.camera_path
        observed_path = self._path_parameters(self.camera_path)

        # Trayectoria causal suavizada. No requiere leer el video completo.
        self.smoothed_path += self.alpha * (observed_path - self.smoothed_path)
        desired_path = self._rigid_matrix(*self.smoothed_path)

        # Lleva el frame crudo actual desde P_t hacia la trayectoria S_t.
        correction = desired_path @ np.linalg.inv(self.camera_path)
        correction = self._zoom_matrix(width, height) @ correction

        stabilized = cv2.warpAffine(
            frame,
            correction[:2, :],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        self.diagnostics["correction_x"] = float(correction[0, 2])
        self.diagnostics["correction_y"] = float(correction[1, 2])
        self.previous_gray = current_gray
        self.frame_index += 1
        return stabilized
