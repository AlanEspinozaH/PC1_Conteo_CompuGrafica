import argparse
import csv
import json
from pathlib import Path

import cv2

from preprocessing import preprocess_motion
from detector import detect_moving_objects
from stabilizer import VideoStabilizer
from tracker import CentroidTracker
from counter import LineCounter


# Configuración básica.

BASE_DIR = Path(__file__).resolve().parent

OFFICIAL_VIDEO_PATH = BASE_DIR / "video" / "conteo_pc1_grafica.mp4"

DEFAULT_VIDEO_PATH = (
    OFFICIAL_VIDEO_PATH
    if OFFICIAL_VIDEO_PATH.exists()
    else BASE_DIR / "video" / "video.mp4"
)

DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Parámetros de detección.

MIN_AREA = 1200


# Estabilización del video.

USE_STABILIZATION = True

STABILIZATION_WINDOW = 30

STABILIZATION_BORDER_SCALE = 1.02


# Parámetros del tracker.

MAX_DISTANCE = 60

MAX_DISAPPEARED = 10


# Parámetros del contador.

LINE_MARGIN = 10

MIN_SEEN_FRAMES = 5


# "down":
# arriba -> abajo = ENTRADA
#
# "up":
# abajo -> arriba = ENTRADA

ENTRY_DIRECTION = "down"


# Argumentos reproducibles en Windows y Ubuntu.

def build_argument_parser():

    parser = argparse.ArgumentParser(
        description="Conteo de cruces de regiones móviles en un video."
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Ruta del video de entrada."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta para eventos.csv y video_resultado.mp4."
    )

    roi_group = parser.add_mutually_exclusive_group()

    roi_group.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="ROI guardada. Si se omite, se selecciona con el mouse."
    )

    roi_group.add_argument(
        "--roi-file",
        type=Path,
        help=(
            "Archivo JSON de ROI. Si no existe, permite seleccionarla y "
            "la guarda; si existe, la reutiliza."
        )
    )

    parser.add_argument(
        "--select-roi-only",
        action="store_true",
        help="Selecciona/guarda la ROI y termina sin procesar el video."
    )

    parser.add_argument(
        "--no-stabilization",
        action="store_true",
        help="Desactiva la estabilización para comparar resultados."
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="No abre ventanas; requiere --roi o un --roi-file existente."
    )

    parser.add_argument(
        "--stabilization-window",
        type=int,
        default=STABILIZATION_WINDOW,
        help="Ventana equivalente del suavizado de trayectoria."
    )

    parser.add_argument(
        "--border-scale",
        type=float,
        default=STABILIZATION_BORDER_SCALE,
        help="Zoom entre 1.0 y 1.20 para ocultar bordes estabilizados."
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=MIN_AREA,
        help="Área mínima de una región móvil; evita editar main.py."
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Detiene una prueba después de esta cantidad de frames."
    )

    return parser


def load_roi(roi_path):

    with roi_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    try:
        return tuple(
            int(data[key])
            for key in ("x", "y", "width", "height")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Formato de ROI inválido en {roi_path}"
        ) from error


def save_roi(roi_path, roi):

    x, y, w, h = map(int, roi)
    roi_path.parent.mkdir(parents=True, exist_ok=True)

    with roi_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            },
            file,
            indent=2
        )
        file.write("\n")


def select_scaled_roi(frame, max_width=1200, max_height=800):
    """Selecciona una ROI visible incluso si el video es vertical 1080x1920."""

    height, width = frame.shape[:2]
    scale = min(
        1.0,
        max_width / width,
        max_height / height
    )

    if scale < 1.0:
        preview = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )
    else:
        preview = frame

    preview_roi = cv2.selectROI(
        "Seleccionar ROI",
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    if scale == 1.0:
        return preview_roi

    return tuple(
        int(round(value / scale))
        for value in preview_roi
    )


# Función principal.

def main():

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.min_area <= 0:
        parser.error("--min-area debe ser mayor que cero")

    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames debe ser mayor que cero")

    roi_path = (
        args.roi_file.expanduser().resolve()
        if args.roi_file is not None
        else None
    )

    has_saved_roi = roi_path is not None and roi_path.is_file()

    if args.no_display and args.roi is None and not has_saved_roi:
        parser.error(
            "--no-display requiere --roi o un --roi-file existente"
        )

    if args.select_roi_only and args.no_display and not has_saved_roi:
        parser.error(
            "--select-roi-only necesita una ventana o un --roi-file existente"
        )

    video_path = args.video.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    events_path = output_dir / "eventos.csv"
    result_video_path = output_dir / "video_resultado.mp4"
    use_stabilization = USE_STABILIZATION and not args.no_stabilization

    # Crea la carpeta de salida.

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # Abre el video.

    cap = cv2.VideoCapture(
        str(video_path)
    )


    if not cap.isOpened():

        print("[ERROR] No se pudo abrir el video:")
        print(video_path)

        return


    # Lee datos del video.

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    duration = 0

    if fps > 0:

        duration = (
            total_frames / fps
        )


    # Si FPS no puede obtenerse,
    # usamos 30 FPS para el video de salida

    output_fps = fps if fps > 0 else 30.0


    print("======================================")
    print(" INFORMACIÓN DEL VIDEO")
    print("======================================")

    print(
        f"Ruta          : {video_path}"
    )

    print(
        f"Resolución    : {width} x {height}"
    )

    print(
        f"FPS           : {fps:.2f}"
    )

    print(
        f"Total frames  : {total_frames}"
    )

    print(
        f"Duración      : {duration:.2f} segundos"
    )

    print(
        f"Estabilización: {use_stabilization}"
    )

    print(
        f"Área mínima   : {args.min_area}"
    )

    print("======================================")


    # Crea el estabilizador.

    stabilizer = VideoStabilizer(
        smoothing_window=args.stabilization_window,
        border_scale=args.border_scale
    )


    # Crea el tracker.

    tracker = CentroidTracker(
        max_distance=MAX_DISTANCE,
        max_disappeared=MAX_DISAPPEARED
    )


    # Lee el primer frame.

    ret, first_frame = cap.read()


    if not ret:

        print(
            "[ERROR] No se pudo leer el primer frame."
        )

        cap.release()

        return


    # Estabiliza el primer frame.

    if use_stabilization:

        first_frame = stabilizer.stabilize(
            first_frame
        )


    # Selecciona la zona de interés.

    print()
    print("======================================")
    print(" SELECCIÓN DE ROI")
    print("======================================")
    print("Selecciona la zona por donde")
    print("pasarán las personas.")
    print()
    print("ENTER o SPACE : confirmar")
    print("C              : cancelar")
    print("======================================")
    print()


    if args.roi is not None:

        roi = tuple(args.roi)

    elif has_saved_roi:

        try:
            roi = load_roi(roi_path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"[ERROR] {error}")
            cap.release()
            return

        print(f"[INFO] ROI cargada desde: {roi_path}")

    else:

        roi = select_scaled_roi(first_frame)

        cv2.destroyWindow(
            "Seleccionar ROI"
        )


    x, y, w, h = map(
        int,
        roi
    )


    # Verifica que la ROI sea válida.

    if (
        w <= 0
        or h <= 0
        or x < 0
        or y < 0
        or x + w > width
        or y + h > height
    ):

        print(
            "[ERROR] ROI inválida."
        )

        cap.release()

        cv2.destroyAllWindows()

        return


    print("======================================")

    if use_stabilization:

        stabilizer.set_exclusion_roi(
            (x, y, w, h),
            first_frame.shape
        )
    print(" ROI")
    print("======================================")

    print(f"x     : {x}")
    print(f"y     : {y}")
    print(f"ancho : {w}")
    print(f"alto  : {h}")

    print("======================================")

    if roi_path is not None and not has_saved_roi:

        save_roi(roi_path, (x, y, w, h))
        print(f"[INFO] ROI guardada en: {roi_path}")

    if args.select_roi_only:

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Selección de ROI finalizada.")
        return


    # Prepara el video de salida.

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    video_writer = cv2.VideoWriter(
        str(result_video_path),
        fourcc,
        output_fps,
        (width, height)
    )


    if not video_writer.isOpened():

        print(
            "[ERROR] No se pudo crear "
            "el video de salida."
        )

        cap.release()

        cv2.destroyAllWindows()

        return


    print(
        f"[INFO] Video de salida: "
        f"{result_video_path}"
    )


    # Configura la línea de conteo.

    # Línea horizontal en la mitad de la ROI

    LINE_Y = h // 2


    counter = LineCounter(
        line_y=LINE_Y,
        margin=LINE_MARGIN,
        min_seen_frames=MIN_SEEN_FRAMES,
        entry_direction=ENTRY_DIRECTION
    )


    print(
        f"[INFO] Línea de conteo Y: {LINE_Y}"
    )

    print(
        f"[INFO] Dirección entrada: "
        f"{ENTRY_DIRECTION}"
    )


    # Guarda la ROI inicial.

    previous_roi = first_frame[
        y:y + h,
        x:x + w
    ].copy()


    # Variables del loop.

    frame_number = 1

    event_records = []


    # Loop principal.

    while True:

        if (
            args.max_frames is not None
            and frame_number >= args.max_frames
        ):
            print(
                f"[INFO] Prueba limitada a {args.max_frames} frames."
            )
            break

        # Lee el siguiente frame.

        ret, frame = cap.read()


        if not ret:

            print(
                "[INFO] Fin del video."
            )

            break


        frame_number += 1

        if args.no_display and frame_number % 300 == 0:

            percentage = (
                100.0 * frame_number / total_frames
                if total_frames > 0
                else 0.0
            )

            print(
                f"[PROGRESO] Frame {frame_number}/{total_frames} "
                f"({percentage:.1f} %)"
            )


        # Guarda el frame original.

        original_frame = frame.copy()


        # Estabiliza el frame.

        if use_stabilization:

            frame = stabilizer.stabilize(
                frame
            )


        # Extrae la ROI.

        current_roi = frame[
            y:y + h,
            x:x + w
        ].copy()


        # Preprocesa para detectar movimiento.

        (
            difference,
            threshold,
            median,
            dilated

        ) = preprocess_motion(
            previous_roi,
            current_roi
        )


        # Detecta objetos en movimiento.

        detections = detect_moving_objects(
            dilated,
            min_area=args.min_area
        )


        # Hace el seguimiento.

        tracked_objects = tracker.update(
            detections
        )


        # Cuenta los cruces.

        events = counter.update(
            tracked_objects
        )


        # Guarda los eventos.

        for event in events:

            timestamp = (
                frame_number / fps
                if fps > 0
                else 0
            )


            event_records.append({

                "frame":
                    frame_number,

                "tiempo_segundos":
                    round(timestamp, 2),

                "id":
                    event["id"],

                "direccion":
                    event["direction"]
            })


            print(
                f"[CONTEO] "
                f"Frame {frame_number} | "
                f"ID {event['id']} | "
                f"{event['direction']} | "
                f"{timestamp:.2f} s"
            )


        # Dibuja la ROI.

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "ROI",
            (
                x,
                max(y - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # Dibuja la línea.

        global_line_y = (
            y + LINE_Y
        )


        cv2.line(
            frame,
            (
                x,
                global_line_y
            ),
            (
                x + w,
                global_line_y
            ),
            (0, 0, 255),
            2
        )


        cv2.putText(
            frame,
            "LINEA DE CONTEO",
            (
                x + 10,
                max(
                    global_line_y - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )


        # Dibuja detecciones.

        for detection in detections:

            dx = detection["x"]
            dy = detection["y"]

            dw = detection["w"]
            dh = detection["h"]

            cx = detection["cx"]
            cy = detection["cy"]

            area = detection["area"]


            # ------------------------------------------------
            # Coordenadas ROI -> Frame completo
            # ------------------------------------------------

            global_x = (
                x + dx
            )

            global_y = (
                y + dy
            )

            global_cx = (
                x + cx
            )

            global_cy = (
                y + cy
            )


            # ------------------------------------------------
            # Bounding Box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (
                    global_x,
                    global_y
                ),
                (
                    global_x + dw,
                    global_y + dh
                ),
                (0, 255, 255),
                2
            )


            # ------------------------------------------------
            # Centroide de detección
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    global_cx,
                    global_cy
                ),
                4,
                (0, 0, 255),
                -1
            )


            # ------------------------------------------------
            # Área
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"Area: {int(area)}",
                (
                    global_x,
                    max(
                        global_y - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1
            )


        # Dibuja tracking.

        for object_id, centroid in tracked_objects.items():

            cx, cy = centroid


            global_cx = (
                x + cx
            )

            global_cy = (
                y + cy
            )


            # ------------------------------------------------
            # Centroide del tracker
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    global_cx,
                    global_cy
                ),
                7,
                (255, 0, 255),
                -1
            )


            # ------------------------------------------------
            # ID
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"ID {object_id}",
                (
                    global_cx - 25,
                    global_cy - 15
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2
            )


        # Muestra datos generales.

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Regiones: {len(detections)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Objetos activos: {len(tracked_objects)}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )


        cv2.putText(
            frame,
            f"Area minima: {args.min_area}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )


        # Muestra estado de estabilización.

        stabilization_text = (
            "Estabilizacion: ON"
            if use_stabilization
            else "Estabilizacion: OFF"
        )


        cv2.putText(
            frame,
            stabilization_text,
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )


        # Calidad de la estimación global de movimiento.

        if use_stabilization:

            stabilization_quality = stabilizer.diagnostics

            cv2.putText(
                frame,
                (
                    f"Puntos/inliers: "
                    f"{stabilization_quality['tracked_points']}/"
                    f"{stabilization_quality['inliers']}"
                ),
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 0),
                1
            )


        # Muestra contadores.

        cv2.putText(
            frame,
            f"Entradas: {counter.entry_count}",
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Salidas: {counter.exit_count}",
            (20, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Total cruces: {counter.get_total()}",
            (20, 265),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )


        # Guarda el frame final.

        video_writer.write(
            frame
        )


        # Muestra ventanas.

        if not args.no_display:

            cv2.imshow(
                "1 - Video original",
                original_frame
            )


            cv2.imshow(
                "2 - Sistema de Conteo",
                frame
            )


            cv2.imshow(
                "3 - Movimiento dilatado",
                dilated
            )


        # Actualiza la ROI anterior.

        previous_roi = (
            current_roi.copy()
        )


        # Controla la salida.

        key = -1

        if not args.no_display:

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


        if (
            key == 27
            or key == ord("q")
        ):

            print(
                "[INFO] Ejecución detenida."
            )

            break


    # Guarda eventos en CSV.

    with open(
        events_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "tiempo_segundos",
                "id",
                "direccion"
            ]
        )


        writer.writeheader()

        writer.writerows(
            event_records
        )


    # Libera recursos.

    cap.release()

    video_writer.release()

    cv2.destroyAllWindows()


    # Muestra resultados finales.

    print()
    print("======================================")
    print(" RESULTADOS FINALES")
    print("======================================")

    print(
        f"Entradas     : {counter.entry_count}"
    )

    print(
        f"Salidas      : {counter.exit_count}"
    )

    print(
        f"Total cruces : {counter.get_total()}"
    )

    print()
    print(
        f"CSV          : {events_path}"
    )

    print(
        f"Video final  : {result_video_path}"
    )

    print("======================================")

    print(
        "[INFO] Programa finalizado correctamente."
    )


# Punto de entrada.

if __name__ == "__main__":

    main()
