import sys, os, glob, cv2, torch
from ultralytics import YOLO
torch.set_num_threads(2)
det = YOLO(sys.argv[1]); cls = YOLO(sys.argv[2]); whole = sys.argv[3] == '1'; TH = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
def flagged(img):
    H, W = img.shape[:2]; r = det.predict(img, imgsz=512, conf=0.20, iou=0.5, max_det=10, verbose=False)[0]
    if len(r.boxes): return True
    if whole:
        p = cls.predict(img, imgsz=224, verbose=False)[0]
        if p.names[p.probs.top1] != 'good' and float(p.probs.top1conf) >= TH: return True
    return False
ev = [l.split() for l in open('/home/claude/ft_eval_list.txt').read().splitlines()]
zp = sum(1 for p, y in ev if y == '1' and flagged(cv2.imread(p))); zf = sum(1 for p, y in ev if y == '0' and flagged(cv2.imread(p)))
C = '/home/claude/textile_dataset/classification/test'; cres = {}
for cl in sorted(os.listdir(C)):
    fs = sorted(glob.glob(f'{C}/{cl}/*'))[:120]; cres[cl] = round(100 * sum(flagged(cv2.imread(p)) for p in fs) / len(fs))
dt = sorted(glob.glob('/home/claude/dtd/test/*.jpg')); fd = sum(flagged(cv2.imread(p)) for p in dt)
pl = [flagged(cv2.imread(p)) for p in ['/tmp/plaid.jpg', '/tmp/plaid2.jpg']]
print(f"RES det={sys.argv[1].split('/')[-3]} cls={sys.argv[2].split('/')[-3]} whole={whole} th={TH}: ZJU defects {zp/4:.1f}% falseAlarm {zf/4:.1f}% | cls-test flagged% {cres} | patterned FA {100*fd/len(dt):.1f}% | plaid {pl}", flush=True)
