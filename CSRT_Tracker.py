from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ==== PATHS ===============================================================
downloads   = Path.home() / "Downloads"
VIDEO_PATH  = downloads / "videos" / "person4.mp4"
OUTPUT_PATH = downloads / "person4_CSRT_final_output.mp4"

# ==== HYPER‑PARAMETERS ====================================================
CONF_THRESHOLD  = 0.05     # YOLO first‑frame confidence
IOU_THRESH      = 0.50     # skip duplicate tracks if IoU > this
HIST_BINS       = (16,16,16)
MAX_LOST_FRAMES = 60       # keep predicting this many missed frames

# --------------------------------------------------------------------------
def extract_hsv_hist(patch, bins=HIST_BINS):
    hsv  = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,bins,[0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def iou(a, b):
    xa,ya = max(a[0],b[0]), max(a[1],b[1])
    xb,yb = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0, xb-xa) * max(0, yb-ya)
    if inter == 0: return 0.0
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter)

# --------------------------------------------------------------------------
class Track:
    def __init__(self, bbox_xywh, label, frame):
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox_xywh)

        x,y,w,h = map(int, bbox_xywh)
        self.bbox  = [x,y,w,h]          # stored as [x,y,w,h]
        self.label = label
        self.lost  = 0                  # consecutive lost frames

        self.feature = extract_hsv_hist(frame[y:y+h, x:x+w])

        # Kalman state (cx,cy,vx,vy)
        cx, cy = x + w/2, y + h/2
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.transitionMatrix   = np.array(
            [[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.measurementMatrix  = np.eye(2,4, dtype=np.float32)
        self.kalman.processNoiseCov    = np.eye(4, dtype=np.float32)*1e-2
        self.kalman.measurementNoiseCov= np.eye(2, dtype=np.float32)*1e-1
        self.kalman.statePre  = np.array([[cx],[cy],[0],[0]], np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()

    # convenience helpers
    def predict_center(self):
        p = self.kalman.predict()
        return float(p[0,0]), float(p[1,0])

    def correct(self, cx, cy):
        self.kalman.correct(np.array([[cx],[cy]], np.float32))

    def box_xyxy(self):
        x,y,w,h = self.bbox
        return [x, y, x+w, y+h]

# --------------------------------------------------------------------------
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(str(VIDEO_PATH))
assert cap.isOpened(), f"Cannot open {VIDEO_PATH}"
W, H   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(str(OUTPUT_PATH),
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps_in, (W,H))

tracks      = []
prev_gray   = None
prev_pts    = None
frame_idx   = 0
start_time  = time.perf_counter()

feature_params = dict(maxCorners=200, qualityLevel=0.3,
                      minDistance=7, blockSize=7)
lk_params = dict(winSize=(15,15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

# ------------------------------- MAIN LOOP ---------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -- First frame: detect & spawn CSRT trackers --------------------------
    if frame_idx == 1:
        det = model.predict(frame, conf=CONF_THRESHOLD,
                            classes=[0,3], verbose=False)[0]
        for box, cls in zip(det.boxes.xyxy.cpu().numpy().astype(int),
                            det.boxes.cls.cpu().numpy().astype(int)):
            if any(tr.label == model.names[int(cls)] and
                   iou(box, tr.box_xyxy()) > IOU_THRESH for tr in tracks):
                continue
            x1,y1,x2,y2 = box
            tracks.append(Track((x1,y1,x2-x1,y2-y1),
                                model.names[int(cls)], frame))
        prev_gray = gray.copy()
        prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                            **feature_params)
    # -- Subsequent frames --------------------------------------------------
    else:
        # 1) global motion compensation
        curr_pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray,
                                                  prev_pts, None, **lk_params)
        good_prev = prev_pts[st.flatten()==1]
        good_curr = curr_pts[st.flatten()==1]
        M = np.eye(2,3, dtype=np.float32)
        if len(good_prev) >= 6:
            M,_ = cv2.estimateAffinePartial2D(good_prev, good_curr,
                                              method=cv2.RANSAC)
        stab = cv2.warpAffine(frame, M, (W,H))

        # 2) update each track
        new_tracks = []
        for tr in tracks:
            pcx, pcy = tr.predict_center()
            ok, bbox = tr.tracker.update(stab)
            if ok:
                tr.lost = 0
                tr.bbox = bbox
                x,y,w,h = map(int,bbox)
                tr.correct(x + w/2, y + h/2)
            else:
                tr.lost += 1
                if tr.lost > MAX_LOST_FRAMES:
                    continue  # give up on this track
                # fabricate a bbox around predicted center
                w,h = tr.bbox[2], tr.bbox[3]
                x,y = int(pcx - w/2), int(pcy - h/2)
                tr.bbox = [x,y,w,h]

            # draw back onto original (unstabilised) frame
            x,y,w,h = tr.bbox
            pts  = np.array([[x,y], [x+w,y+h]], np.float32).reshape(-1,1,2)
            invM = cv2.invertAffineTransform(M)
            ox1,oy1, ox2,oy2 = cv2.transform(pts, invM).reshape(-1,2).flatten()
            cv2.rectangle(frame, (int(ox1),int(oy1)),
                          (int(ox2),int(oy2)), (0,255,0), 2)
            cv2.putText(frame, tr.label, (int(ox1),int(oy1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.circle(frame, (int(pcx),int(pcy)), 4, (0,0,255), -1)
            new_tracks.append(tr)
        tracks = new_tracks

    # -- FPS overlay --------------------------------------------------------
    fps = 1.0 / (time.perf_counter() - t0)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0,255,255), 2)

    out.write(frame)
    prev_gray = gray.copy()
    prev_pts  = cv2.goodFeaturesToTrack(prev_gray, mask=None,
                                        **feature_params)

# ------------------------------- WRAP‑UP ----------------------------------
total = time.perf_counter() - start_time
print(f"Processed {frame_idx} frames in {total:.2f}s — "
      f"Avg FPS: {frame_idx/total:.2f}")
print(f"Output saved to: {OUTPUT_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()

