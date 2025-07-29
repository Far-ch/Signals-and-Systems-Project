import cv2
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
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
VIDEO_PATH = str(Path.home()/"Downloads"/"videos"/"person2.mp4")
OUTPUT_PATH = str(Path.home()/"Downloads"/"person2_frcnn_kjf_output.mp4")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_TH = 0.5
MIN_LK_POINTS = 3
REINIT_KP = 2
DROP_LIMIT = 500
GRID = (4, 4)
MAX_CORNERS = [4000, 3000]
MIN_DIST = [3, 3]
REID_MAX_AGE = 60
REID_SIM_THRESH = 0.30
HOG_SIM_THRESH = 0.6
DEEP_SIM_THRESH = 0.85  # Cosine similarity for deep features (higher = more similar)

# ----------- Deep feature extractor (ResNet-18, remove classifier) -----------
deep_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
deep_model.fc = torch.nn.Identity()  # Remove the last layer
deep_model = deep_model.eval().to(DEVICE)
deep_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
def extract_deep_feature(img, box):
    x1, y1, x2, y2 = [int(e) for e in box]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 16 or patch.shape[1] < 16:
        return None
    try:
        inp = deep_transform(patch).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = deep_model(inp).cpu().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat
    except Exception:
        return None

def deep_sim(f1, f2):
    if f1 is None or f2 is None or len(f1) != len(f2): return 0
    return float(np.dot(f1, f2) / (np.linalg.norm(f1)+1e-8) / (np.linalg.norm(f2)+1e-8))

# ----------- Color histogram -----------
def color_hist(img, box):
    x1, y1, x2, y2 = [int(e) for e in box]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0 or patch.shape[0] < 5 or patch.shape[1] < 5:
        return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], None, [8,8,8], [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()
def hist_sim(h1, h2):
    if h1 is None or h2 is None: return 1.0
    return cv2.compareHist(h1.reshape(8,8,8), h2.reshape(8,8,8), cv2.HISTCMP_BHATTACHARYYA)

# ----------- HOG descriptor -----------
def compute_hog(img, box):
    x1, y1, x2, y2 = [int(e) for e in box]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(img.shape[1]-1, x2); y2 = min(img.shape[0]-1, y2)
    patch = img[y1:y2, x1:x2]
    WIN_W, WIN_H = 64, 128
    if patch.size == 0 or patch.shape[0] < 16 or patch.shape[1] < 16:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (WIN_W, WIN_H))
    hog = cv2.HOGDescriptor(_winSize=(WIN_W, WIN_H),
                            _blockSize=(16,16),
                            _blockStride=(8,8),
                            _cellSize=(8,8),
                            _nbins=9)
    desc = hog.compute(resized)
    return desc.flatten() if desc is not None else None
def hog_sim(h1, h2):
    if h1 is None or h2 is None or len(h1) != len(h2): return 0
    h1 = h1 / (np.linalg.norm(h1) + 1e-8)
    h2 = h2 / (np.linalg.norm(h2) + 1e-8)
    return float(np.dot(h1, h2))

# ----------- Kalman Filter Class -----------
class KalmanBox:
    count = 0
    def __init__(self, box, hist, hog_desc, deep_feat):
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
        # Memory for appearance
        self.hist = hist
        self.hog_desc = hog_desc
        self.deep_feat = deep_feat
        self.memory = [hist] if hist is not None else []
        self.memory_hog = [hog_desc] if hog_desc is not None else []
        self.memory_deep = [deep_feat] if deep_feat is not None else []
        self.max_mem = 5
        self.age = 0
        self.last_frame = 0
        self.last_box = box

    def _to_box(self, s):
        cx,cy,w,h = s[0],s[1],s[6],s[7]
        return (int(cx-w/2), int(cy-h/2), int(cx+w/2), int(cy+h/2))
    def predict(self):
        s = self.kf.predict().flatten()
        self.lost += 1
        self.last = self._to_box(s)
        return self.last
    def update(self, box, hist=None, hog_desc=None, deep_feat=None):
        x1,y1,x2,y2 = box
        cx,cy = (x1+x2)/2,(y1+y2)/2
        w,h   = x2-x1,    y2-y1
        meas  = np.array([cx,cy,w,h],np.float32).reshape(-1,1)
        self.kf.correct(meas)
        self.lost = 0
        self.last = self._to_box(self.kf.statePost.flatten())
        self.last_box = box
        if hist is not None:
            self.hist = hist
            self.memory.append(hist)
            if len(self.memory) > self.max_mem:
                self.memory.pop(0)
        if hog_desc is not None:
            self.hog_desc = hog_desc
            self.memory_hog.append(hog_desc)
            if len(self.memory_hog) > self.max_mem:
                self.memory_hog.pop(0)
        if deep_feat is not None:
            self.deep_feat = deep_feat
            self.memory_deep.append(deep_feat)
            if len(self.memory_deep) > self.max_mem:
                self.memory_deep.pop(0)
    def get_hist(self):
        if len(self.memory) == 0: return self.hist
        return np.mean(self.memory, axis=0)
    def get_hog(self):
        if len(self.memory_hog) == 0: return self.hog_desc
        return np.mean(self.memory_hog, axis=0)
    def get_deep(self):
        if len(self.memory_deep) == 0: return self.deep_feat
        return np.mean(self.memory_deep, axis=0)

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
    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame_bgr, (x1,y1-th-8),(x1+tw,y1), (0,255,0), -1)
    cv2.putText(frame_bgr, label, (x1,y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
out.write(frame_bgr)

gray0 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
trackers = []
lost_tracks = []  # memory for lost tracks

for i, (b, cid) in enumerate(zip(boxes, labels)):
    hist = color_hist(frame_bgr, b)
    hog = compute_hog(frame_bgr, b)
    deep = extract_deep_feature(frame_bgr, b)
    kf = KalmanBox(tuple(b), hist, hog, deep)
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
        'age': 0,
        'missed': 0
    })

old_gray = gray0.copy()
frame_count = 1
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

    active_ids = set()
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

        if p1 is not None and st.sum() >= MIN_LK_POINTS:
            newp = p1[st==1].reshape(-1,2)
            oldp = obj['p0'][st==1].reshape(-1,2)
            pts  = newp[filter_pts(newp,oldp)]
            if len(pts) >= MIN_LK_POINTS:
                obj['history'].append(pts)
                allpts = np.vstack(obj['history'])
                hull   = cv2.convexHull(allpts.astype(np.float32))
                bx,by,bw,bh = cv2.boundingRect(hull)
                hist = color_hist(frame, (bx,by,bx+bw,by+bh))
                hog  = compute_hog(frame, (bx,by,bx+bw,by+bh))
                deep = extract_deep_feature(frame, (bx,by,bx+bw,by+bh))
                kf.update((bx,by,bx+bw,by+bh), hist, hog, deep)
                x1, y1, x2, y2 = bx, by, bx+bw, by+bh
                obj['p0'] = pts.reshape(-1,1,2)
                obj['lost'] = 0
                obj['missed'] = 0
                active_ids.add(kf.id)
            else:
                x1,y1,x2,y2 = kf.predict()
                obj['p0'], obj['lost'] = None, obj['lost']+1
                obj['missed'] += 1
        else:
            x1,y1,x2,y2 = kf.predict()
            obj['p0'], obj['lost'] = None, obj['lost']+1
            obj['missed'] += 1

        # Wider search window for keypoint reinit
        if obj['lost'] == REINIT_KP:
            pad = 20  # search outside previous box
            xx1,yy1 = max(0,x1-pad), max(0,y1-pad)
            xx2,yy2 = min(w0,x2+pad), min(h0,y2+pad)
            newkp = get_distributed_kps(
                gray[yy1:yy2,xx1:xx2],
                GRID, 2000, 5
            )
            if newkp is not None and len(newkp) >= MIN_LK_POINTS:
                newkp[:,:,0]+=xx1; newkp[:,:,1]+=yy1
                obj['history'].clear()
                obj['history'].append(newkp.reshape(-1,2))
                obj['p0'], obj['lost'] = newkp, 0

        if obj['lost'] >= DROP_LIMIT:
            lost_tracks.append({
                'hist': kf.get_hist(),
                'hog': kf.get_hog(),
                'deep': kf.get_deep(),
                'cls': obj['cls'],
                'kf': kf,
                'last_box': (x1,y1,x2,y2),
                'last_seen': frame_count
            })
            trackers.remove(obj)
            continue

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
            x1, y1, x2, y2 = kf.last
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            label = CLASSES.get(obj['cls'], str(obj['cls']))
            cv2.putText(frame, f"{label} #{obj['kf'].id} age:{obj['age']}", (x1, y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # ------- Improved matching for lost track recovery -------
    if len(lost_tracks) > 0:
        for lt in lost_tracks[:]:
            if frame_count - lt['last_seen'] > REID_MAX_AGE:
                lost_tracks.remove(lt)
                continue
            hist_lost = lt['hist']
            hog_lost = lt['hog']
            deep_lost = lt['deep']
            best_obj = None
            best_score = 0
            for obj in trackers:
                if obj['kf'].id in active_ids:
                    continue
                hist_obj = obj['kf'].get_hist()
                hog_obj = obj['kf'].get_hog()
                deep_obj = obj['kf'].get_deep()
                sim_hist = 1.0 - hist_sim(hist_lost, hist_obj)
                sim_hog = hog_sim(hog_lost, hog_obj)
                sim_deep = deep_sim(deep_lost, deep_obj)
                # Use weighted vote of all features
                total_sim = 0.4*sim_hist + 0.3*sim_hog + 0.3*sim_deep
                if sim_hist > (1-REID_SIM_THRESH) and sim_hog > HOG_SIM_THRESH and sim_deep > DEEP_SIM_THRESH and lt['cls'] == obj['cls']:
                    if total_sim > best_score:
                        best_obj = obj
                        best_score = total_sim
            if best_obj is not None:
                best_obj['lost'] = 0
                best_obj['missed'] = 0
                best_obj['kf'].hist = hist_lost
                best_obj['kf'].hog_desc = hog_lost
                best_obj['kf'].deep_feat = deep_lost
                best_obj['kf'].memory.append(hist_lost)
                best_obj['kf'].memory_hog.append(hog_lost)
                best_obj['kf'].memory_deep.append(deep_lost)
                lost_tracks.remove(lt)
                print(f"Re-identified track #{best_obj['kf'].id} at frame {frame_count}")

    out.write(frame)
    old_gray = gray.copy()

cap.release()
out.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
fps = frame_count / total_time
print(f"✅ Done – saved to {OUTPUT_PATH}")
print(f"Average FPS: {fps:.2f}")
