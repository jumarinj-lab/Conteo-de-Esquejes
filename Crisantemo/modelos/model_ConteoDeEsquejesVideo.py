import cv2
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import cvzone
import supervision as sv  # pip install supervision

# =========================
# CONFIG (VIDEO)
# =========================
VIDEO_PATH = "prueba2gimbal.mp4"
PT_MODEL_PATH = "best26V2.pt"
OV_DIR = "best26V2_openvino_model"
IMGSZ = 640  # OpenVINO actual espera 640x640

RESIZE_W, RESIZE_H = 960, 480
RECT_Y1 = 80
COUNT_LINE_Y = 350

CONF = 0.25
IOU = 0.50
FRAME_STRIDE = 1

DRAW_BOXES = True
DRAW_POINTS = True

# =========================
# 1) Export OpenVINO si no existe y cargar
# =========================
if not Path(OV_DIR).exists():
    print(f"[INFO] No existe {OV_DIR}/. Exportando a OpenVINO (imgsz={IMGSZ})...")
    YOLO(PT_MODEL_PATH).export(format="openvino", imgsz=IMGSZ)
    print("[INFO] Export OpenVINO terminado.")

print(f"[INFO] Cargando modelo OpenVINO desde: {OV_DIR}/")
model = YOLO(OV_DIR, task="detect")

# =========================
# 2) Video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("No pude abrir el video. Revisa ruta/códec.")

fps_in = cap.get(cv2.CAP_PROP_FPS)
fps_in = fps_in if fps_in and fps_in > 1 else 30.0
print(f"[INFO] Video abierto @ {fps_in:.2f} FPS")

# =========================
# 3) Tracker (ByteTrack)
# =========================
tracker = sv.ByteTrack(
    track_activation_threshold=CONF,
    lost_track_buffer=int(fps_in * 1.0),
    minimum_matching_threshold=0.8
)

# Conteo esquejes
last_center = {}
counted_ids = set()
count = 0

# =========================
# Loop
# =========================
cv2.namedWindow("VIEW", cv2.WINDOW_NORMAL)

fps_smooth = 0.0
t_prev = time.time()
frame_idx = 0

print("[INFO] Presiona ESC para salir.")

while True:
    ok, frame = cap.read()
    if not ok:
        print("[INFO] Fin del video.")
        break

    frame_idx += 1
    if frame_idx % FRAME_STRIDE != 0:
        continue

    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    RECT_X1 = 0
    RECT_X2 = frame.shape[1]
    ROI_Y2 = frame.shape[0]

    roi = frame[RECT_Y1:ROI_Y2, RECT_X1:RECT_X2]
    if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
        cv2.imshow("VIEW", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("[INFO] Video detenido por el usuario.")
            break
        continue

    # Detección
    results = model.predict(
        roi,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        verbose=False
    )

    xyxy = np.empty((0, 4), dtype=np.float32)
    confs = np.empty((0,), dtype=np.float32)
    class_ids = np.empty((0,), dtype=np.int32)

    if results and results[0].boxes is not None and len(results[0].boxes) > 0:
        b = results[0].boxes
        xyxy = b.xyxy.cpu().numpy() if hasattr(b.xyxy, "cpu") else np.array(b.xyxy)
        confs = b.conf.cpu().numpy() if hasattr(b.conf, "cpu") else np.array(b.conf)
        class_ids = b.cls.cpu().numpy().astype(int) if hasattr(b.cls, "cpu") else np.array(b.cls).astype(int)

        xyxy = xyxy.astype(np.float32)
        confs = confs.astype(np.float32)

    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confs,
        class_id=class_ids
    )

    detections = tracker.update_with_detections(detections)

    # ROI y línea roja
    cv2.rectangle(
        frame,
        (RECT_X1, RECT_Y1),
        (RECT_X2, ROI_Y2),
        (0, 255, 255),
        2
    )

    cv2.line(
        frame,
        (RECT_X1, COUNT_LINE_Y),
        (RECT_X2, COUNT_LINE_Y),
        (0, 0, 255),
        2
    )

    # Conteo esquejes
    for i in range(len(detections)):
        tid = int(detections.tracker_id[i])
        x1, y1, x2, y2 = detections.xyxy[i].astype(int)

        x1f, y1f = x1 + RECT_X1, y1 + RECT_Y1
        x2f, y2f = x2 + RECT_X1, y2 + RECT_Y1

        cx = (x1f + x2f) // 2
        cy = (y1f + y2f) // 2

        inside_x = (RECT_X1 <= cx < RECT_X2)

        if tid in last_center:
            prev_cx, prev_cy = last_center[tid]
            crossed_down = (prev_cy < COUNT_LINE_Y <= cy)

            if crossed_down and inside_x and (tid not in counted_ids):
                count += 1
                counted_ids.add(tid)
                cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)

        last_center[tid] = (cx, cy)

        if DRAW_POINTS:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        if DRAW_BOXES:
            cv2.rectangle(
                frame,
                (x1f, y1f),
                (x2f, y2f),
                (0, 255, 0),
                2
            )

    # FPS
    t_now = time.time()
    dt = max(t_now - t_prev, 1e-6)
    t_prev = t_now
    inst_fps = 1.0 / dt
    fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps

    # UI
    cvzone.putTextRect(frame, f"CONTADOR: {count}", (30, 40), 2, 2)
    cvzone.putTextRect(frame, f"FPS: {fps_smooth:.1f}", (30, 90), 2, 2)

    cv2.imshow("VIEW", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("[INFO] Video detenido por el usuario.")
        break

cap.release()
cv2.destroyAllWindows()

# =========================
# RESULTADO FINAL EN CONSOLA
# =========================
print("\n==============================")
print(f"TOTAL DE ESQUEJES CONTADOS EN EL VIDEO: {count}")
print("==============================\n")