# KFC_tracker
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from pathlib import Path
import uuid
import time  # For FPS calculation

# ==== PATHS ====
downloads = Path.home() / "Downloads"
VIDEO_PATH = downloads / "videos" / "person4.mp4"
OUTPUT_PATH = downloads / "person4_kcf_final_output.mp4"

# ==== CONSTANTS ====
CONF_TH = 0.6
MAX_LOST = 180
HIST_THRESHOLD = 0.5
HIST_BINS = (16, 16, 16)

# ==== COCO LABELS ====
COCO = {i: name for i, name in enumerate([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'])}

def get_histogram(patch, bins=HIST_BINS):
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def compare_hist(hist1, hist2):
    return cv2.compareHist(hist1.astype(np.float32), hist2.astype(np.float32), cv2.HISTCMP_CORREL)

class StrongKalmanFilter:
    def __init__(self, x, y):
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = 1.0
        self.kf.measurementMatrix = np.zeros((4, 8), np.float32)
        for i in range(4):
            self.kf.measurementMatrix[i, i] = 1.0
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0], [0], [0], [0], [0]], dtype=np.float32)

    def predict(self):
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    def correct(self, x, y):
        self.kf.correct(np.array([[x], [y], [0], [0]], dtype=np.float32))

# ==== MODEL ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.to(device).eval()
transform = transforms.Compose([transforms.ToTensor()])

# ==== VIDEO ====
cap = cv2.VideoCapture(str(VIDEO_PATH))
assert cap.isOpened(), f"Could not open {VIDEO_PATH}"
W, H = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out = cv2.VideoWriter(str(OUTPUT_PATH), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

tracks = []
frame_idx = 0
t0 = prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if frame_idx == 1:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = transform(rgb).to(device)
        with torch.no_grad():
            preds = model([img_tensor])[0]

        boxes = preds['boxes'].cpu().numpy()
        labels_raw = preds['labels'].cpu().numpy()
        scores = preds['scores'].cpu().numpy()
        keep = scores >= CONF_TH
        boxes, labels_raw = boxes[keep], labels_raw[keep]

        for box, cls_id in zip(boxes, labels_raw):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, (x1, y1, w, h))
            hist = get_histogram(patch)
            kalman = StrongKalmanFilter(x1 + w//2, y1 + h//2)
            tracks.append({
                "id": str(uuid.uuid4())[:8],
                "tracker": tracker,
                "label": COCO.get(int(cls_id), f"class{cls_id}"),
                "hist": hist,
                "lost": 0,
                "kalman": kalman
            })
    else:
        new_tracks = []
        for tr in tracks:
            ok, bbox = tr["tracker"].update(frame)
            x, y, w, h = map(int, bbox)
            x2, y2 = x + w, y + h

            if not ok or x < 0 or y < 0 or x2 > W or y2 > H:
                tr["lost"] += 1
                pred_x, pred_y = tr["kalman"].predict()
                if tr["lost"] <= MAX_LOST:
                    cv2.circle(frame, (pred_x, pred_y), 4, (0, 0, 255), -1)
                    new_tracks.append(tr)
                continue

            patch = frame[y:y+h, x:x+w]
            if patch.size > 0:
                hist_new = get_histogram(patch)
                similarity = compare_hist(tr["hist"], hist_new)
                if similarity < HIST_THRESHOLD:
                    tr["lost"] += 1
                    if tr["lost"] <= MAX_LOST:
                        new_tracks.append(tr)
                    continue
                else:
                    tr["lost"] = 0
                    tr["hist"] = 0.8 * tr["hist"] + 0.2 * hist_new
                    tr["kalman"].correct(x + w//2, y + h//2)

            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, tr["label"], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            new_tracks.append(tr)

        tracks = new_tracks

    # ==== FPS Overlay ====
    now = time.time()
    fps_now = 1.0 / (now - prev + 1e-6)
    prev = now
    cv2.putText(frame, f"FPS: {fps_now:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    out.write(frame)

# ==== Done ====
cap.release()
out.release()
total_time = time.time() - t0
avg_fps = frame_idx / total_time
print(f"✅ Saved to: {OUTPUT_PATH}")
print(f"Average FPS: {avg_fps:.2f}")
