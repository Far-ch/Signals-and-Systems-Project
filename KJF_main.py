import cv2
import numpy as np
import torch
from torchvision import models, transforms
from collections import deque
from pathlib import Path
import time

# ----------- COCO classes -----------
CLASSES = {
    0:"background", 1:"person", 2:"bicycle", 3:"car", 4:"motorcycle", 5:"airplane",
    6:"bus",7:"train",8:"truck",9:"boat",10:"trafficlight",11:"firehydrant",
    13:"bench",14:"bird",15:"cat",16:"dog",17:"horse",18:"sheep",19:"cow",
    20:"elephant",21:"bear",22:"zebra",23:"giraffe",24:"backpack",25:"umbrella",
    27:"handbag",28:"tie",31:"snowboard",32:"sportsball",33:"kite",34:"baseballbat",
    35:"skis",36:"skateboard",37:"surfboard",38:"tennisracket",39:"bottle",
    40:"wineglass",41:"cup",42:"fork",43:"knife",44:"spoon",45:"bowl",46:"banana",
    47:"apple",48:"sandwich",49:"orange",50:"broccoli",51:"carrot",52:"hotdog",
    53:"pizza",54:"donut",55:"cake",56:"chair",57:"couch",58:"pottedplant",59:"bed",
    60:"diningtable",61:"toilet",62:"tv",63:"laptop",64:"mouse",65:"remote",
    66:"keyboard",67:"cellphone",68:"microwave",69:"oven",70:"toaster",71:"sink",
    72:"refrigerator",73:"book",74:"clock",75:"vase",76:"scissors",77:"teddybear",
    78:"hairdrier",79:"toothbrush"
}

# ----------- Settings -----------
VIDEO_PATH = str(Path.home()/"Downloads"/"videos"/"person4.mp4")
OUTPUT_PATH = str(Path.home()/"Downloads"/"person4_frcnn_kjf_output.mp4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_TH = 0.5         # Detection confidence threshold
MIN_LK_POINTS = 3     # Minimum good LK points to trust tracking
REINIT_KP = 2        # Frames before reinit keypoints
DROP_LIMIT = 500      # Drop track if lost too long
GRID = (4, 4)
MAX_CORNERS = [4000, 3000]
MIN_DIST = [3, 3]

# ----------- Kalman Filter Class -----------
class KalmanBox:
    count = 0
    def __init__(self, box):
        self.id = KalmanBox.count; KalmanBox.count += 1
        self.lost = 0
        self.kf = cv2.KalmanFilter(10, 4)
        M = np.zeros((4,10), np.float32)
        M[[0,1,2,3],[0,1,6,7]] = 1
        self.kf.measurementMatrix = M
        dt, dt2 = 1/30.0, 0.5*(1/30.0)**2
        T = np.eye(10, dtype=np.float32)
        T[0,2],T[0,4]=dt,dt2; T[1,3],T[1,5]=dt,dt2
        T[2,4],T[3,5]=dt,dt;   T[6,8],T[7,9]=dt,dt
        self.kf.transitionMatrix    = T
        self.kf.processNoiseCov     = np.eye(10, dtype=np.float32)*0.01
        self.kf.measurementNoiseCov = np.eye(4,  dtype=np.float32)*0.02
        x1,y1,x2,y2 = box
        cx,cy = (x1+x2)/2,(y1+y2)/2
        w,h   = x2-x1,    y2-y1
        self.kf.statePost = np.array([cx,cy,0,0,0,0,w,h,0,0],np.float32).reshape(-1,1)
        self.last = self._to_box(self.kf.statePost.flatten())
    def _to_box(self, s):
        cx,cy,w,h = s[0],s[1],s[6],s[7]
        return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))
    def predict(self):
        s = self.kf.predict().flatten()
        self.lost += 1
        self.last = self._to_box(s)
        return self.last
    def update(self, box):
        x1,y1,x2,y2 = box
        cx,cy = (x1+x2)/2,(y1+y2)/2
        w,h   = x2-x1,    y2-y1
        meas  = np.array([cx,cy,w,h],np.float32).reshape(-1,1)
        self.kf.correct(meas)
        self.lost = 0
        self.last = self._to_box(self.kf.statePost.flatten())

# ----------- Helper Functions -----------
def get_distributed_kps(gray, grid, max_c, min_d):
    h,w=gray.shape; gh,gw=grid; pts=[]
    for i in range(gh):
        for j in range(gw):
            roi=gray[i*h//gh:(i+1)*h//gh, j*w//gw:(j+1)*w//gw]
            c=cv2.goodFeaturesToTrack(roi,
                maxCorners=max_c//(gh*gw),
                qualityLevel=0.05, minDistance=min_d)
            if c is not None:
                for [[x,y]] in c:
                    pts.append([x+j*w//gw, y+i*h//gh])
    return np.array(pts,np.float32).reshape(-1,1,2) if pts else None

def filter_pts(nw,old,max_m=25,thr=2.08,sk=100):
    if len(nw)==0 or len(old)==0: return np.array([],bool)
    mv=np.linalg.norm(nw-old,axis=1)<max_m
    d =np.linalg.norm(nw-nw.mean(0),axis=1)
    cl=d<=thr*np.mean(np.delete(d,0))
    sp=np.linalg.norm(nw-old,axis=1)
    sd=(sp>=sp.mean()-sk*sp.std())&(sp<=sp.mean()+sk*sp.std())
    return mv & cl & sd

# ----------- Main Script -----------
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame_bgr = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")

frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
model = models.detection.fasterrcnn_resnet50_fpn(
    weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
).to(DEVICE).eval()
transform = transforms.Compose([transforms.ToTensor()])
img_t = transform(frame_rgb).to(DEVICE)

with torch.no_grad():
    preds = model([img_t])[0]

boxes  = preds["boxes"].cpu().numpy().astype(int)
labels = preds["labels"].cpu().numpy()
scores = preds["scores"].cpu().numpy()

keep = scores >= CONF_TH
boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

h0, w0 = frame_bgr.shape[:2]
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    cap.get(cv2.CAP_PROP_FPS),
    (w0, h0)
)
# Draw detection boxes on first frame
for (x1,y1,x2,y2), cls_id, sc in zip(boxes, labels, scores):
    name = CLASSES.get(int(cls_id), str(int(cls_id)))
    label = f"{name} {sc:.2f}"
    cv2.rectangle(frame_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    (tw,th),_ = cv2.getTextSize(label,
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr,
                  (x1,y1-th-8),(x1+tw,y1),
                  (0,255,0), -1)
    cv2.putText(frame_bgr, label,
                (x1,y1-4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,0,0), 2)
out.write(frame_bgr)

gray0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
trackers = []
for i, (b, cid) in enumerate(zip(boxes, labels)):
    kf = KalmanBox(tuple(b))
    x1,y1,x2,y2 = b
    kp = get_distributed_kps(
        gray0[y1:y2, x1:x2],
        GRID,
        MAX_CORNERS[min(i,len(MAX_CORNERS)-1)],
        MIN_DIST   [min(i,len(MIN_DIST)-1)]
    )
    if kp is not None and len(kp) >= MIN_LK_POINTS:
        kp[:,:,0] += x1; kp[:,:,1] += y1
        history = deque([kp.reshape(-1,2)], maxlen=REINIT_KP)
    else:
        kp, history = None, deque([], maxlen=REINIT_KP)
    trackers.append({
        'kf': kf,
        'p0': kp,
        'history': history,
        'cls': int(cid),
        'lost': 0,
        'age': 0   # age field for track memory
    })

old_gray = gray0.copy()
frame_count = 1
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    for obj in trackers[:]:
        kf = obj['kf']
        x1,y1,x2,y2 = kf.last
        obj['age'] += 1

        # LK Optical Flow
        if obj['p0'] is not None:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, gray, obj['p0'], None,
                winSize=(15,15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS|
                          cv2.TERM_CRITERIA_COUNT,10,0.03)
            )
        else:
            p1, st = None, 0

        # If enough points: update box tightly to hull
        if p1 is not None and st.sum() >= MIN_LK_POINTS:
            newp = p1[st==1].reshape(-1,2)
            oldp = obj['p0'][st==1].reshape(-1,2)
            pts  = newp[filter_pts(newp,oldp)]
            if len(pts) >= MIN_LK_POINTS:
                obj['history'].append(pts)
                allpts = np.vstack(obj['history'])
                hull   = cv2.convexHull(allpts.astype(np.float32))
                bx,by,bw,bh = cv2.boundingRect(hull)
                # Update Kalman from the hull-based box
                kf.update((bx,by,bx+bw,by+bh))
                x1, y1, x2, y2 = bx, by, bx+bw, by+bh
                obj['p0'] = pts.reshape(-1,1,2)
                obj['lost'] = 0
            else:
                x1,y1,x2,y2 = kf.predict()
                obj['p0'], obj['lost'] = None, obj['lost']+1
        else:
            x1,y1,x2,y2 = kf.predict()
            obj['p0'], obj['lost'] = None, obj['lost']+1

        # Re-init keypoints after several lost frames
        if obj['lost'] == REINIT_KP:
            xx1,yy1 = max(0,x1), max(0,y1)
            xx2,yy2 = min(w0,x2), min(h0,y2)
            newkp = get_distributed_kps(
                gray[yy1:yy2,xx1:xx2],
                GRID, 2000, 5
            )
            if newkp is not None and len(newkp) >= MIN_LK_POINTS:
                newkp[:,:,0]+=xx1; newkp[:,:,1]+=yy1
                obj['history'].clear()
                obj['history'].append(newkp.reshape(-1,2))
                obj['p0'], obj['lost'] = newkp, 0

        # Drop track if lost for too long
        if obj['lost'] >= DROP_LIMIT:
            trackers.remove(obj)
            continue

        # Draw TIGHT bounding box always around the object
        color = (0,255,0)
        if obj['p0'] is not None and len(obj['p0']) >= MIN_LK_POINTS:
            pts = obj['p0'].reshape(-1,2)
            hull = cv2.convexHull(pts.astype(np.float32))
            bx, by, bw, bh = cv2.boundingRect(hull)
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), color, 2)
            label = CLASSES.get(obj['cls'], str(obj['cls']))
            cv2.putText(frame, f"{label} #{obj['kf'].id} age:{obj['age']}", (bx, by-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            for pt in obj['p0']:
                cv2.circle(frame, tuple(pt[0].astype(int)), 2, (0,0,255), -1)
        else:
            # Draw predicted box if no points
            x1, y1, x2, y2 = kf.last
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            label = CLASSES.get(obj['cls'], str(obj['cls']))
            cv2.putText(frame, f"{label} #{obj['kf'].id} age:{obj['age']}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    out.write(frame)
    old_gray = gray.copy()

cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
fps = frame_count / total_time
print(f"✅ Done – saved to {OUTPUT_PATH}")
print(f"Average FPS: {fps:.2f}")
