import cv2
import csv
from pathlib import Path

from preprocessing import preprocess_motion
from detector import detect_moving_objects
from stabilizer import VideoStabilizer
from tracker import CentroidTracker
from counter import LineCounter


# Configuración básica.

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "video" / "video2.mp4"

OUTPUT_DIR = BASE_DIR / "output"

EVENTS_PATH = OUTPUT_DIR / "eventos.csv"

RESULT_VIDEO_PATH = OUTPUT_DIR / "video_resultado.mp4"


# Parámetros de detección.

MIN_AREA = 1200


# Estabilización del video.

USE_STABILIZATION = True


# Parámetros del tracker.

MAX_DISTANCE = 60

MAX_DISAPPEARED = 10


# Parámetros del contador.

LINE_MARGIN = 10

MIN_SEEN_FRAMES = 5


ENTRY_DIRECTION = "right"


# Función principal.

def main():

    # Crea la carpeta de salida.

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Abre el video.

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )


    if not cap.isOpened():

        print("[ERROR] No se pudo abrir el video:")
        print(VIDEO_PATH)

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
        f"Ruta          : {VIDEO_PATH}"
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
        f"Estabilización: {USE_STABILIZATION}"
    )

    print(
        f"Área mínima   : {MIN_AREA}"
    )

    print("======================================")


    # Crea el estabilizador.

    stabilizer = VideoStabilizer()


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

    if USE_STABILIZATION:

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


    roi = cv2.selectROI(
        "Seleccionar ROI",
        first_frame,
        showCrosshair=True,
        fromCenter=False
    )


    cv2.destroyWindow(
        "Seleccionar ROI"
    )


    x, y, w, h = map(
        int,
        roi
    )


    # Verifica que la ROI sea válida.

    if w == 0 or h == 0:

        print(
            "[ERROR] ROI inválida."
        )

        cap.release()

        cv2.destroyAllWindows()

        return


    print("======================================")
    print(" ROI")
    print("======================================")

    print(f"x     : {x}")
    print(f"y     : {y}")
    print(f"ancho : {w}")
    print(f"alto  : {h}")

    print("======================================")


    # Prepara el video de salida.

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )


    video_writer = cv2.VideoWriter(
        str(RESULT_VIDEO_PATH),
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
        f"{RESULT_VIDEO_PATH}"
    )


    # Configura la línea de conteo.

    LINE_X = w // 2


    counter = LineCounter(
        line_x=LINE_X,
        margin=LINE_MARGIN,
        min_seen_frames=MIN_SEEN_FRAMES,
        entry_direction=ENTRY_DIRECTION
    )


    print(
        f"[INFO] Línea de conteo X: {LINE_X}"
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

        # Lee el siguiente frame.

        ret, frame = cap.read()


        if not ret:

            print(
                "[INFO] Fin del video."
            )

            break


        frame_number += 1


        # Guarda el frame original.

        original_frame = frame.copy()


        # Estabiliza el frame.

        if USE_STABILIZATION:

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
            min_area=MIN_AREA
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

        global_line_x = x + LINE_X

        cv2.line(
            frame,
            (global_line_x, y),
            (global_line_x, y + h),
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "LINEA DE CONTEO",
            (
                global_line_x + 10,
                max(y + 20, 20)
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
            f"Area minima: {MIN_AREA}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2
        )


        # Muestra estado de estabilización.

        stabilization_text = (
            "Estabilizacion: ON"
            if USE_STABILIZATION
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


        # Muestra contadores.

        cv2.putText(
            frame,
            f"Entradas: {counter.entry_count}",
            (20, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Salidas: {counter.exit_count}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Total cruces: {counter.get_total()}",
            (20, 250),
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
        EVENTS_PATH,
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
        f"CSV          : {EVENTS_PATH}"
    )

    print(
        f"Video final  : {RESULT_VIDEO_PATH}"
    )

    print("======================================")

    print(
        "[INFO] Programa finalizado correctamente."
    )


# Punto de entrada.

if __name__ == "__main__":

    main()