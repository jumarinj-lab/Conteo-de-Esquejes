import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cvzone

# =========================
# CONFIG VIDEO / MODELO
# =========================
VIDEO_PATH = "prueba2gimbal.mp4"
PT_MODEL_PATH = "best26V2.pt"
OV_DIR = "best26V2_openvino_model"
IMGSZ = 640

RESIZE_W, RESIZE_H = 960, 480

# Mismo largo vertical del recuadro de interés del modelo original
RECT_Y1 = 80
ROI_Y2 = RESIZE_H

# Línea guía de conteo, igual al modelo original
COUNT_LINE_Y = 350

CONF = 0.25
IOU = 0.50
FRAME_STRIDE = 1

# =========================
# CONFIG ROI DINÁMICO
# =========================
DEFAULT_RECT_X1 = 0
DEFAULT_RECT_X2 = RESIZE_W

PADDING_X = 70
MIN_BED_WIDTH = 250
MAX_BED_WIDTH = RESIZE_W

SMOOTH_ALPHA = 0.85
CENTER_TOLERANCE_PX = 60

# El modelo detecta en segundo plano, pero no dibuja cajas de esquejes
DRAW_DETECTIONS = False

# Mostrar punto medio de la nube de detecciones
DRAW_CLOUD_CENTER = True


# =========================
# FUNCIÓN: estimar cama desde nube de detecciones
# =========================
def estimate_bed_from_detections(
    xyxy_full,
    frame_width,
    prev_x1=None,
    prev_x2=None
):
    """
    Usa las detecciones del modelo para estimar la nube de esquejes
    y, con eso, la posición lateral de la cama.

    xyxy_full: cajas en coordenadas del frame completo.
    """

    # Si hay pocas detecciones, conservar el último ROI estable
    if xyxy_full is None or len(xyxy_full) < 3:
        if prev_x1 is not None and prev_x2 is not None:
            return prev_x1, prev_x2, False
        return DEFAULT_RECT_X1, DEFAULT_RECT_X2, False

    x1_boxes = xyxy_full[:, 0]
    x2_boxes = xyxy_full[:, 2]
    cx_boxes = (x1_boxes + x2_boxes) / 2

    # Percentiles para evitar que detecciones falsas dañen el recuadro
    left_limit = int(np.percentile(cx_boxes, 5))
    right_limit = int(np.percentile(cx_boxes, 95))

    min_box_x = int(np.percentile(x1_boxes, 5))
    max_box_x = int(np.percentile(x2_boxes, 95))

    detected_x1 = min(left_limit, min_box_x) - PADDING_X
    detected_x2 = max(right_limit, max_box_x) + PADDING_X

    # Limitar al tamaño de la imagen
    detected_x1 = max(0, detected_x1)
    detected_x2 = min(frame_width, detected_x2)

    bed_width = detected_x2 - detected_x1

    # Validar ancho
    if bed_width < MIN_BED_WIDTH or bed_width > MAX_BED_WIDTH:
        if prev_x1 is not None and prev_x2 is not None:
            return prev_x1, prev_x2, False
        return DEFAULT_RECT_X1, DEFAULT_RECT_X2, False

    # Suavizado temporal para evitar saltos bruscos del recuadro
    if prev_x1 is not None and prev_x2 is not None:
        detected_x1 = int(SMOOTH_ALPHA * prev_x1 + (1 - SMOOTH_ALPHA) * detected_x1)
        detected_x2 = int(SMOOTH_ALPHA * prev_x2 + (1 - SMOOTH_ALPHA) * detected_x2)

    return detected_x1, detected_x2, True


# =========================
# 1) Exportar OpenVINO si no existe
# =========================
if not Path(OV_DIR).exists():
    print(f"[INFO] No existe {OV_DIR}/. Exportando a OpenVINO imgsz={IMGSZ}...")
    YOLO(PT_MODEL_PATH).export(format="openvino", imgsz=IMGSZ)
    print("[INFO] Export OpenVINO terminado.")

print(f"[INFO] Cargando modelo OpenVINO desde: {OV_DIR}/")
model = YOLO(OV_DIR, task="detect")


# =========================
# 2) Abrir video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("No pude abrir el video. Revisa ruta o códec.")

fps_in = cap.get(cv2.CAP_PROP_FPS)
fps_in = fps_in if fps_in and fps_in > 1 else 30.0

print(f"[INFO] Video abierto @ {fps_in:.2f} FPS")
print("[INFO] Presiona ESC para salir.")


# =========================
# LOOP PRINCIPAL
# =========================
cv2.namedWindow("VIEW", cv2.WINDOW_NORMAL)

fps_smooth = 0.0
t_prev = time.time()
frame_idx = 0

prev_bed_x1 = None
prev_bed_x2 = None

while True:
    ok, frame = cap.read()

    if not ok:
        print("[INFO] Fin del video.")
        break

    frame_idx += 1

    if frame_idx % FRAME_STRIDE != 0:
        continue

    # Redimensionar para procesamiento estable
    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
    h, w = frame.shape[:2]

    # =========================
    # ROI vertical fijo
    # =========================
    roi_vertical = frame[RECT_Y1:ROI_Y2, 0:w]

    if roi_vertical.size == 0:
        cv2.imshow("VIEW", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] Video detenido por el usuario.")
            break

        continue

    # =========================
    # Detección en segundo plano
    # =========================
    results = model.predict(
        roi_vertical,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    xyxy_full = np.empty((0, 4), dtype=np.float32)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        b = results[0].boxes

        xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.array(b.xyxy)
        xyxy = xyxy.astype(np.float32)

        # Convertir coordenadas del ROI vertical a coordenadas del frame completo
        xyxy_full = xyxy.copy()
        xyxy_full[:, 1] += RECT_Y1
        xyxy_full[:, 3] += RECT_Y1

    # =========================
    # Estimar cama desde nube de esquejes
    # =========================
    bed_x1, bed_x2, valid_bed = estimate_bed_from_detections(
        xyxy_full,
        frame_width=w,
        prev_x1=prev_bed_x1,
        prev_x2=prev_bed_x2
    )

    prev_bed_x1 = bed_x1
    prev_bed_x2 = bed_x2

    bed_center_x = (bed_x1 + bed_x2) // 2
    image_center_x = w // 2

    error_x = bed_center_x - image_center_x
    bed_width = bed_x2 - bed_x1

    # =========================
    # Estado para guiar encuadre
    # =========================
    if abs(error_x) <= CENTER_TOLERANCE_PX:
        status = "CAMA CENTRADA"
        status_color = (0, 255, 0)
        command = "MANTENER"
    elif error_x > CENTER_TOLERANCE_PX:
        status = "CAMA A LA DERECHA"
        status_color = (0, 165, 255)
        command = "CORREGIR DERECHA"
    else:
        status = "CAMA A LA IZQUIERDA"
        status_color = (0, 165, 255)
        command = "CORREGIR IZQUIERDA"

    # =========================
    # Dibujar asistencia visual
    # =========================
    roi_color = (0, 255, 255) if valid_bed else (0, 165, 255)

    # Recuadro dinámico de interés
    cv2.rectangle(
        frame,
        (bed_x1, RECT_Y1),
        (bed_x2, ROI_Y2),
        roi_color,
        3
    )

    # Bordes laterales estimados de la cama
    cv2.line(
        frame,
        (bed_x1, RECT_Y1),
        (bed_x1, ROI_Y2),
        (255, 0, 255),
        2
    )

    cv2.line(
        frame,
        (bed_x2, RECT_Y1),
        (bed_x2, ROI_Y2),
        (255, 0, 255),
        2
    )

    # Línea guía de conteo
    cv2.line(
        frame,
        (bed_x1, COUNT_LINE_Y),
        (bed_x2, COUNT_LINE_Y),
        (0, 0, 255),
        3
    )

    cvzone.putTextRect(
        frame,
        "LINEA DE CONTEO",
        (max(bed_x1 + 10, 10), COUNT_LINE_Y - 15),
        1.1,
        2,
        colorR=(0, 0, 255)
    )

    # Centro de la imagen
    cv2.line(
        frame,
        (image_center_x, RECT_Y1),
        (image_center_x, ROI_Y2),
        (255, 255, 255),
        1
    )

    # Centro estimado de la cama
    cv2.line(
        frame,
        (bed_center_x, RECT_Y1),
        (bed_center_x, ROI_Y2),
        (0, 255, 0),
        2
    )

    # Punto central de la nube de detecciones, no muestra esqueje por esqueje
    if DRAW_CLOUD_CENTER and len(xyxy_full) >= 3:
        xs_centers = (xyxy_full[:, 0] + xyxy_full[:, 2]) / 2
        ys_centers = (xyxy_full[:, 1] + xyxy_full[:, 3]) / 2

        cloud_x = int(np.mean(xs_centers))
        cloud_y = int(np.mean(ys_centers))

        cv2.circle(
            frame,
            (cloud_x, cloud_y),
            8,
            (0, 255, 255),
            -1
        )

    # Opcional: si algún día quieres ver las cajas, activa DRAW_DETECTIONS = True
    if DRAW_DETECTIONS and len(xyxy_full) > 0:
        for box in xyxy_full:
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                1
            )

    # =========================
    # FPS
    # =========================
    t_now = time.time()
    dt = max(t_now - t_prev, 1e-6)
    t_prev = t_now

    inst_fps = 1.0 / dt
    fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

    # =========================
    # UI limpia
    # =========================
    cvzone.putTextRect(
        frame,
        f"FPS: {fps_smooth:.1f}",
        (30, 40),
        1.5,
        2
    )

    cvzone.putTextRect(
        frame,
        f"DETECCIONES USADAS: {len(xyxy_full)}",
        (30, 85),
        1.4,
        2
    )

    cvzone.putTextRect(
        frame,
        f"ANCHO ROI: {bed_width}px",
        (30, 130),
        1.4,
        2
    )

    cvzone.putTextRect(
        frame,
        f"ERROR X: {error_x}px",
        (30, 175),
        1.4,
        2
    )

    cvzone.putTextRect(
        frame,
        status,
        (30, 225),
        1.4,
        2,
        colorR=status_color
    )

    cvzone.putTextRect(
        frame,
        command,
        (30, 275),
        1.5,
        2
    )

    cvzone.putTextRect(
        frame,
        "GRABE HASTA QUE EL FINAL DE LA CAMA CRUCE LA LINEA ROJA",
        (30, 325),
        1.0,
        2,
        colorR=(0, 0, 255)
    )

    # =========================
    # Mostrar
    # =========================
    cv2.imshow("VIEW", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("[INFO] Video detenido por el usuario.")
        break


# =========================
# Cierre
# =========================
cap.release()
cv2.destroyAllWindows()

print("\n==============================")
print("[INFO] Prueba de encuadre finalizada.")
print("==============================\n")