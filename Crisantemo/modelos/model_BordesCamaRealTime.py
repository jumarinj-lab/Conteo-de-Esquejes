import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cvzone

BASE_DIR = Path(__file__).resolve().parent

# =========================
# CONFIG MODELO DE SEGMENTACIÓN
# =========================
PT_MODEL_PATH = BASE_DIR / "best (2).pt"   # Modelo de segmentacion de cama
OV_MODEL_DIR = BASE_DIR / "best (2)_openvino_model"
IMGSZ = 640

# =========================
# CONFIG VIDEO
# =========================
VIDEO_PATH = BASE_DIR / "prueba2gimbal.mp4"

# Resolución de trabajo
RESIZE_W, RESIZE_H = 960, 480

# =========================
# CONFIG VISUAL / ROI
# =========================
RECT_Y1 = 80
ROI_Y2 = RESIZE_H

# Línea guía del modelo de conteo original
COUNT_LINE_Y = 350

# Segmentación
CONF = 0.25
IOU = 0.50
FRAME_STRIDE = 1
MAX_DETECTIONS = 5

# Tolerancia para saber si la cama está centrada
CENTER_TOLERANCE_PX = 60

# Suavizado temporal del recuadro
SMOOTH_ALPHA = 0.85

# Margen alrededor de la cama segmentada
PADDING_X = 20
PADDING_Y = 10

# Área mínima para aceptar una máscara como cama
MIN_MASK_AREA = 3000

# Mostrar máscara semitransparente
DRAW_MASK = True

# Mostrar contorno de la cama
DRAW_CONTOUR = True

# Guardar video procesado
SAVE_OUTPUT = True
OUTPUT_PATH = BASE_DIR / "salida_segmentacion_cama_video.mp4"


# =========================
# FUNCIÓN: suavizar valores
# =========================
def smooth_value(prev, new, alpha=0.85):
    if prev is None:
        return new
    return int(alpha * prev + (1 - alpha) * new)


# =========================
# FUNCIÓN: obtener máscara principal
# =========================
def get_largest_segmentation_mask(result, frame_shape):
    """
    Toma el resultado de YOLO segmentación y devuelve:
    - mask_uint8: máscara binaria de la cama
    - contour: contorno principal
    - area: área del contorno
    """

    h, w = frame_shape[:2]

    if result.masks is None or result.masks.xy is None or len(result.masks.xy) == 0:
        return None, None, 0

    best_contour = None
    best_area = 0

    for polygon in result.masks.xy:
        if polygon is None or len(polygon) < 3:
            continue

        pts = np.array(polygon, dtype=np.int32)
        area = cv2.contourArea(pts)

        if area > best_area:
            best_area = area
            best_contour = pts

    if best_contour is None or best_area < MIN_MASK_AREA:
        return None, None, best_area

    mask_uint8 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_uint8, [best_contour], 255)

    return mask_uint8, best_contour, best_area


# =========================
# FUNCIÓN: calcular ROI desde máscara
# =========================
def estimate_roi_from_mask(
    mask,
    frame_width,
    frame_height,
    prev_x1=None,
    prev_x2=None,
    prev_y1=None,
    prev_y2=None
):
    """
    Calcula el recuadro dinámico a partir de la máscara segmentada.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return prev_x1, prev_x2, prev_y1, prev_y2, False

    x1 = int(np.percentile(xs, 2)) - PADDING_X
    x2 = int(np.percentile(xs, 98)) + PADDING_X

    y1 = int(np.percentile(ys, 2)) - PADDING_Y
    y2 = int(np.percentile(ys, 98)) + PADDING_Y

    x1 = max(0, x1)
    x2 = min(frame_width, x2)

    y1 = max(RECT_Y1, y1)
    y2 = min(frame_height, y2)

    if x2 <= x1 or y2 <= y1:
        return prev_x1, prev_x2, prev_y1, prev_y2, False

    # Suavizado temporal
    x1 = smooth_value(prev_x1, x1, SMOOTH_ALPHA)
    x2 = smooth_value(prev_x2, x2, SMOOTH_ALPHA)
    y1 = smooth_value(prev_y1, y1, SMOOTH_ALPHA)
    y2 = smooth_value(prev_y2, y2, SMOOTH_ALPHA)

    return x1, x2, y1, y2, True


# =========================
# 1) CARGAR MODELO OPENVINO
# =========================
if not OV_MODEL_DIR.exists():
    if not PT_MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo .pt para exportar: {PT_MODEL_PATH}")

    print(f"[INFO] No existe OpenVINO: {OV_MODEL_DIR}")
    print(f"[INFO] Exportando segmentacion a OpenVINO con imgsz={IMGSZ}...")
    YOLO(str(PT_MODEL_PATH), task="segment").export(format="openvino", imgsz=IMGSZ)

if not OV_MODEL_DIR.exists():
    raise FileNotFoundError(f"No se encontro la carpeta OpenVINO exportada: {OV_MODEL_DIR}")

print(f"[INFO] Cargando modelo OpenVINO desde: {OV_MODEL_DIR}")
model = YOLO(str(OV_MODEL_DIR), task="segment")

print("[INFO] Calentando inferencia OpenVINO...")
_warmup_frame = np.zeros((RESIZE_H, RESIZE_W, 3), dtype=np.uint8)
model.predict(_warmup_frame, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DETECTIONS, verbose=False)


# =========================
# 2) ABRIR VIDEO
# =========================
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("No pude abrir el video. Revisa la ruta o el códec.")

fps_in = cap.get(cv2.CAP_PROP_FPS)
fps_in = fps_in if fps_in and fps_in > 1 else 30.0

print(f"[INFO] Video abierto @ {fps_in:.2f} FPS")
print("[INFO] Presiona ESC para salir.")


# =========================
# 3) CONFIG GUARDADO DE VIDEO
# =========================
writer = None

if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps_in,
        (RESIZE_W, RESIZE_H)
    )


# =========================
# 4) LOOP PRINCIPAL
# =========================
cv2.namedWindow("VIEW", cv2.WINDOW_NORMAL)

frame_idx = 0
fps_smooth = 0.0
t_prev = time.time()

prev_x1 = None
prev_x2 = None
prev_y1 = None
prev_y2 = None

while True:
    ok, frame = cap.read()

    if not ok:
        print("[INFO] Fin del video.")
        break

    frame_idx += 1

    if frame_idx % FRAME_STRIDE != 0:
        continue

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
    h, w = frame.shape[:2]

    # =========================
    # PREDICCIÓN DE SEGMENTACIÓN
    # =========================
    results = model.predict(
        frame,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        max_det=MAX_DETECTIONS,
        verbose=False
    )

    result = results[0]

    mask, contour, mask_area = get_largest_segmentation_mask(result, frame.shape)

    valid_bed = False

    if mask is not None:
        bed_x1, bed_x2, bed_y1, bed_y2, valid_bed = estimate_roi_from_mask(
            mask,
            frame_width=w,
            frame_height=h,
            prev_x1=prev_x1,
            prev_x2=prev_x2,
            prev_y1=prev_y1,
            prev_y2=prev_y2
        )

        if valid_bed:
            prev_x1, prev_x2 = bed_x1, bed_x2
            prev_y1, prev_y2 = bed_y1, bed_y2

    else:
        # Si no detecta cama, conserva el último recuadro válido
        if prev_x1 is not None and prev_x2 is not None:
            bed_x1, bed_x2 = prev_x1, prev_x2
            bed_y1, bed_y2 = prev_y1, prev_y2
        else:
            bed_x1, bed_x2 = 0, w
            bed_y1, bed_y2 = RECT_Y1, ROI_Y2

    # =========================
    # CENTRO Y ESTADO DE CAMA
    # =========================
    bed_center_x = (bed_x1 + bed_x2) // 2
    image_center_x = w // 2

    error_x = bed_center_x - image_center_x

    if abs(error_x) <= CENTER_TOLERANCE_PX:
        status = "CAMA CENTRADA"
        status_color = (0, 255, 0)
    elif error_x > CENTER_TOLERANCE_PX:
        status = "CAMA A LA DERECHA"
        status_color = (0, 165, 255)
    else:
        status = "CAMA A LA IZQUIERDA"
        status_color = (0, 165, 255)

    # =========================
    # DIBUJAR MÁSCARA
    # =========================
    if DRAW_MASK and mask is not None:
        overlay = frame.copy()

        overlay[mask > 0] = (
            0.6 * overlay[mask > 0] + 0.4 * np.array([0, 255, 0])
        ).astype(np.uint8)

        frame = overlay

    # =========================
    # DIBUJAR CONTORNO
    # =========================
    if DRAW_CONTOUR and contour is not None:
        cv2.polylines(
            frame,
            [contour],
            isClosed=True,
            color=(0, 255, 0),
            thickness=2
        )

    # =========================
    # DIBUJAR ROI DINÁMICO
    # =========================
    roi_color = (0, 255, 255) if valid_bed else (0, 165, 255)

    cv2.rectangle(
        frame,
        (bed_x1, bed_y1),
        (bed_x2, bed_y2),
        roi_color,
        3
    )

    # Bordes laterales
    cv2.line(
        frame,
        (bed_x1, bed_y1),
        (bed_x1, bed_y2),
        (255, 0, 255),
        2
    )

    cv2.line(
        frame,
        (bed_x2, bed_y1),
        (bed_x2, bed_y2),
        (255, 0, 255),
        2
    )

    # Línea roja guía de conteo
    cv2.line(
        frame,
        (bed_x1, COUNT_LINE_Y),
        (bed_x2, COUNT_LINE_Y),
        (0, 0, 255),
        3
    )

    # Centro de imagen
    cv2.line(
        frame,
        (image_center_x, bed_y1),
        (image_center_x, bed_y2),
        (255, 255, 255),
        1
    )

    # Centro estimado de cama
    cv2.line(
        frame,
        (bed_center_x, bed_y1),
        (bed_center_x, bed_y2),
        (0, 255, 0),
        2
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
    # UI MÍNIMA
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
        status,
        (30, 90),
        1.5,
        2,
        colorR=status_color
    )

    # =========================
    # MOSTRAR / GUARDAR
    # =========================
    cv2.imshow("VIEW", frame)

    if SAVE_OUTPUT and writer is not None:
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("[INFO] Video detenido por el usuario.")
        break


# =========================
# CIERRE
# =========================
cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()

print("\n==============================")
print("[INFO] Segmentación de cama en video finalizada.")
if SAVE_OUTPUT:
    print(f"[INFO] Video guardado en: {OUTPUT_PATH}")
print("==============================\n")