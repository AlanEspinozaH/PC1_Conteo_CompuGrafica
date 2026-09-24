"""Compara movimiento geométrico residual con y sin estabilización."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from stabilizer import VideoStabilizer


def estimate_global_motion(first_frame, second_frame):
    """Devuelve un proxy en píxeles de traslación + rotación global."""
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=20,
    )
    if points is None:
        return None

    moved_points, status, _ = cv2.calcOpticalFlowPyrLK(
        first_gray,
        second_gray,
        points,
        None,
    )
    if moved_points is None or status is None:
        return None

    valid = status.reshape(-1) == 1
    previous = points.reshape(-1, 2)[valid]
    current = moved_points.reshape(-1, 2)[valid]
    if len(previous) < 10:
        return None

    matrix, _ = cv2.estimateAffinePartial2D(
        previous,
        current,
        method=cv2.RANSAC,
    )
    if matrix is None:
        return None

    translation = np.hypot(matrix[0, 2], matrix[1, 2])
    angle = abs(np.arctan2(matrix[1, 0], matrix[0, 0]))
    height, width = first_gray.shape
    rotational_displacement = angle * np.hypot(width, height) / 2.0
    return float(translation + rotational_displacement)


def summarize(values):
    return {
        "media": float(np.mean(values)),
        "mediana": float(np.median(values)),
        "percentil_95": float(np.percentile(values, 95)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa movimiento residual antes y después de estabilizar."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--stabilization-window", type=int, default=30)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=300,
        help="Muestra avance cada N frames; use 0 para ocultarlo.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {video_path}")

    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frames = (
        min(video_frames, args.max_frames)
        if args.max_frames is not None and video_frames > 0
        else video_frames
    )

    stabilizer = VideoStabilizer(
        smoothing_window=args.stabilization_window,
        border_scale=1.0,
    )
    previous_raw = None
    previous_stabilized = None
    raw_motion = []
    stabilized_motion = []
    successful_estimations = 0
    processed_frames = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

        stabilized = stabilizer.stabilize(frame)
        successful_estimations += int(stabilizer.diagnostics["estimated"])

        if previous_raw is not None:
            raw_value = estimate_global_motion(previous_raw, frame)
            stabilized_value = estimate_global_motion(
                previous_stabilized,
                stabilized,
            )
            if raw_value is not None and stabilized_value is not None:
                raw_motion.append(raw_value)
                stabilized_motion.append(stabilized_value)

        previous_raw = frame
        previous_stabilized = stabilized
        processed_frames += 1

        if (
            args.progress_every > 0
            and processed_frames % args.progress_every == 0
        ):
            if target_frames > 0:
                percentage = 100.0 * processed_frames / target_frames
                progress = (
                    f"{processed_frames}/{target_frames} "
                    f"({percentage:.1f} %)"
                )
            else:
                progress = str(processed_frames)

            print(f"Procesando frames: {progress}", flush=True)

    capture.release()

    if not raw_motion:
        raise SystemExit("No hubo suficientes pares de frames para medir.")

    raw_summary = summarize(raw_motion)
    stabilized_summary = summarize(stabilized_motion)
    reduction = 100.0 * (
        1.0 - stabilized_summary["mediana"] / raw_summary["mediana"]
    )

    print(f"Video: {video_path}")
    print(f"Frames procesados: {processed_frames}")
    print(f"Pares medidos: {len(raw_motion)}")
    print(f"Transformaciones válidas: {successful_estimations}")
    print()
    print("Movimiento residual aproximado (píxeles):")
    print(
        f"  Sin estabilizar | media={raw_summary['media']:.3f} "
        f"mediana={raw_summary['mediana']:.3f} "
        f"p95={raw_summary['percentil_95']:.3f}"
    )
    print(
        f"  Estabilizado    | media={stabilized_summary['media']:.3f} "
        f"mediana={stabilized_summary['mediana']:.3f} "
        f"p95={stabilized_summary['percentil_95']:.3f}"
    )
    print(f"Reducción de la mediana: {reduction:.1f} %")
    print()
    print(
        "Esta métrica evalúa estabilidad geométrica, no exactitud del conteo. "
        "El CSV todavía debe compararse con la anotación manual."
    )


if __name__ == "__main__":
    main()
