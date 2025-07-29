# mosse_kalman8_tracker.py
import time, uuid
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision import models, transforms

# ==== PATHS ====
downloads = Path.home() / "Downloads"
VIDEO_PATH = downloads / "videos" / "person4.mp4"
OUTPUT_PATH = downloads / "person4_mosse_output.mp4"

# ─── HYPER‑PARAMS ───────────────────────────────────────
CONF_TH       = 0.50   # Fast‑R‑CNN confidence (first frame only)
HIST_THRESH   = 0.65   # HSV similarity gate
IOU_THRESH    = 0.30   # IoU gate with Kalman prediction
MAX_LOST      = 250    # frames tracker may stay lost
HIST_BINS     = (32,32,32)

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

# ─── HSV histogram helpers ─────────────────────────────
def hsv_hist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1,2], None, HIST_BINS, [0,180,0,256,0,256])
    cv2.normalize(h, h)
    return h.flatten()

def hist_sim(h1, h2):
    return cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32), cv2.HISTCMP_CORREL)

def iou(bb1, bb2):
    xA = max(bb1[0], bb2[0]); yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2]); yB = min(bb1[3], bb2[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    a1 = (bb1[2]-bb1[0]) * (bb1[3]-bb1[1])
    a2 = (bb2[2]-bb2[0]) * (bb2[3]-bb2[1])
    return inter / (a1 + a2 - inter)

# ─── 8‑state Kalman (cx,cy,w,h,vx,vy,vw,vh) ────────────
class Kalman8:
    def __init__(self, cx, cy, w, h):
        self.k = cv2.KalmanFilter(8,4)
        F = np.eye(8, dtype=np.float32); F[0,4]=F[1,5]=F[2,6]=F[3,7]=1
        self.k.transitionMatrix = F
        H = np.zeros((4,8), np.float32)
        H[0,0]=H[1,1]=H[2,2]=H[3,3]=1
        self.k.measurementMatrix = H
        self.k.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-3
        self.k.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.k.errorCovPost = np.eye(8, dtype=np.float32)
        self.k.statePost = np.array([[cx],[cy],[w],[h],[0],[0],[0],[0]], dtype=np.float32)

    def predict(self):
        s = self.k.predict()
        return s[0,0], s[1,0], s[2,0], s[3,0]

    def correct(self, cx, cy, w, h):
        self.k.correct(np.array([[cx],[cy],[w],[h]], dtype=np.float32))

# ─── Fast R‑CNN (only first frame) ─────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
det = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()
tfm = transforms.Compose([transforms.ToTensor()])

# ─── Video IO ───────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH); assert cap.isOpened()
W, H = int(cap.get(3)), int(cap.get(4))
fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps_in, (W, H))

tracks = []; fid = 0; t0 = prev = time.time()

# ─── First frame detection ─────────────────────────────
ret, frame = cap.read(); fid += 1; assert ret
with torch.no_grad():
    p = det([tfm(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(device)])[0]

for box, cid, score in zip(p['boxes'], p['labels'], p['scores']):
    if score < CONF_TH: continue
    x1, y1, x2, y2 = map(int, box.cpu().numpy()); w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0: continue
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
    label = COCO.get(int(cid), 'cls')
    cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    mosse = cv2.legacy.TrackerMOSSE_create(); mosse.init(frame, (x1, y1, w, h))
    kf = Kalman8(x1 + w/2, y1 + h/2, w, h)
    tracks.append(dict(id=str(uuid.uuid4())[:8], tracker=mosse, kalman=kf,
                       hist=hsv_hist(frame[y1:y2, x1:x2]),
                       label=label, lost=0))

# ─── Main loop ─────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok: break
    fid += 1; kept = []
    for tr in tracks:
        ok, bb = tr['tracker'].update(frame)
        if ok:
            x, y, w, h = map(int, bb); x2, y2 = x + w, y + h
            if w <= 0 or h <= 0: ok = False
        if ok:
            patch = frame[y:y+h, x:x+w]
            sim = hist_sim(tr['hist'], hsv_hist(patch)) if patch.size else 0
            cxp, cyp, wp, hp = tr['kalman'].predict()
            iou_ok = iou((x, y, x2, y2), (cxp - wp/2, cyp - hp/2, cxp + wp/2, cyp + hp/2)) >= IOU_THRESH
            if sim >= HIST_THRESH and iou_ok:
                tr['hist'] = 0.9 * tr['hist'] + 0.1 * hsv_hist(patch)
                tr['kalman'].correct(x + w/2, y + h/2, w, h)
                tr['lost'] = 0
            else:
                ok = False
        if not ok:
            tr['lost'] += 1
        if tr['lost'] <= MAX_LOST:
            if tr['lost'] == 0:
                cv2.rectangle(frame, (x, y), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, tr['label'], (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
            kept.append(tr)
    tracks = kept

    now = time.time(); fps_now = 1 / (now - prev + 1e-6); prev = now
    cv2.putText(frame, f"FPS:{fps_now:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    out.write(frame)

# ─── Done ──────────────────────────────────────────────
cap.release(); out.release()
print("✅ saved to", OUTPUT_PATH)
print(f"Average FPS: {fid / (time.time() - t0):.2f}")
