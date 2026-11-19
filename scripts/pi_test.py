"""
PS5 Textile Defect Detection - Raspberry Pi test script (Step 11)

Runs the two ExecuTorch models on a folder of images:
  detector   -> finds WHERE each defect is (boxes)
  classifier -> says WHAT each defect is (hole, stain, ...)
  severity   -> Low / Medium / High, and a PASS / REWORK / REJECT decision

Usage (on the Pi, inside the virtual environment):
  python pi_test.py --images demo_test_images
Results: annotated images + results.csv in the "results" folder, speed summary printed at the end.
"""
import argparse, csv, os, time, glob
import cv2
from ultralytics import YOLO

try:
    import psutil
except ImportError:
    psutil = None

# ---------------------------------------------------------------- severity rules
# Defects that weaken the fabric are treated more strictly.
CRITICAL = {"hole_burst", "broken_yarn_needle"}
LEVELS = ["Low", "Medium", "High"]


def severity(defect_type, area_pct):
    """area_pct = box area as % of the whole image."""
    if area_pct < 0.5:
        level = 0
    elif area_pct < 2.0:
        level = 1
    else:
        level = 2
    if defect_type in CRITICAL:
        level = min(level + 1, 2)
    return LEVELS[level]


def decision(severities):
    if not severities:
        return "PASS"
    if "High" in severities:
        return "REJECT"
    if "Medium" in severities:
        return "REWORK"
    return "PASS"


def model_path(p):
    """Ultralytics recognises an ExecuTorch model by a folder name ending in '_executorch_model'.
    The Colab notebook saved them as 'detector_executorch' / 'classifier_executorch',
    so give such a folder the expected name with a link (nothing is copied)."""
    p = os.path.normpath(p)
    if os.path.isdir(p) and not p.endswith("_executorch_model"):
        alias = p + "_executorch_model"
        if not os.path.exists(alias):
            os.symlink(os.path.basename(p), alias)
        return alias
    return p


COLORS = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="models/detector_executorch", help="detector model folder")
    ap.add_argument("--cls", default="models/classifier_executorch", help="classifier model folder")
    ap.add_argument("--images", default="demo_test_images", help="folder of test images")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--conf", type=float, default=0.20, help="detection confidence (lower = finds more defects)")
    ap.add_argument("--max-det", type=int, default=10, help="max defects per image (each one is classified)")
    ap.add_argument("--det-size", type=int, default=512)
    ap.add_argument("--cls-size", type=int, default=224)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    print("Loading models ...")
    det = YOLO(model_path(a.det), task="detect")
    cls = YOLO(model_path(a.cls), task="classify")

    files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp") for f in glob.glob(os.path.join(a.images, f"*.{ext}")))
    if not files:
        raise SystemExit(f"No images found in {a.images}")

    # warm-up (the first run is always slower)
    first = cv2.imread(files[0])
    det.predict(first, imgsz=a.det_size, conf=a.conf, verbose=False)
    cls.predict(first, imgsz=a.cls_size, verbose=False)

    rows, det_times, total_times = [], [], []
    for n, path in enumerate(files, 1):
        img = cv2.imread(path)
        if img is None:
            print("  skip (cannot read):", path)
            continue
        H, W = img.shape[:2]
        t0 = time.perf_counter()
        r = det.predict(img, imgsz=a.det_size, conf=a.conf, max_det=a.max_det, verbose=False)[0]
        t1 = time.perf_counter()

        fs = max(0.3, min(W, H) / 1000)          # text size follows image size
        th = 1 if fs < 0.6 else 2
        sevs, labels, img_rows = [], [], []
        for box, score in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            x0, y0, x1, y1 = box
            bw, bh = x1 - x0, y1 - y0
            # crop with a little fabric around the defect, like the training crops
            px, py = 0.15 * bw, 0.15 * bh
            cx0, cy0 = int(max(0, x0 - px)), int(max(0, y0 - py))
            cx1, cy1 = int(min(W, x1 + px)), int(min(H, y1 + py))
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            c = cls.predict(crop, imgsz=a.cls_size, verbose=False)[0]
            dtype = c.names[c.probs.top1]
            dprob = float(c.probs.top1conf)
            if dtype == "good":           # detector saw something, classifier unsure
                dtype = "unclassified_defect"
            area_pct = 100.0 * bw * bh / (W * H)
            sev = severity(dtype, area_pct)
            sevs.append(sev)
            labels.append(f"{dtype}:{sev}")
            col = COLORS[sev]
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
            cv2.putText(img, f"{dtype} {sev} {score:.2f}", (int(x0), max(14, int(y0) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
            img_rows.append({"image": os.path.basename(path), "defect_type": dtype, "type_conf": round(dprob, 3),
                         "det_conf": round(score, 3), "area_pct": round(area_pct, 3), "severity": sev})
        t2 = time.perf_counter()

        dec = decision(sevs)
        cv2.putText(img, dec, (8, int(30 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.6,
                    (0, 0, 255) if dec == "REJECT" else (0, 165, 255) if dec == "REWORK" else (0, 180, 0), th + 1)
        cv2.imwrite(os.path.join(a.out, os.path.basename(path)), img)
        if not img_rows:
            img_rows.append({"image": os.path.basename(path), "defect_type": "none", "type_conf": "",
                             "det_conf": "", "area_pct": 0, "severity": ""})
        for row in img_rows:
            row["decision"] = dec
        rows.extend(img_rows)
        det_times.append(t1 - t0)
        total_times.append(t2 - t0)
        print(f"[{n}/{len(files)}] {os.path.basename(path)}: {len(sevs)} defect(s) {labels} -> {dec}  "
              f"({1000 * (t2 - t0):.0f} ms)")

    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "defect_type", "type_conf", "det_conf", "area_pct", "severity", "decision"])
        w.writeheader()
        w.writerows(rows)

    avg_det = 1000 * sum(det_times) / len(det_times)
    avg_tot = 1000 * sum(total_times) / len(total_times)
    print("\n========== SUMMARY ==========")
    print(f"Images tested         : {len(total_times)}")
    print(f"Detector only         : {avg_det:.0f} ms per image  ({1000 / avg_det:.1f} FPS)")
    print(f"Detector + classifier : {avg_tot:.0f} ms per image  ({1000 / avg_tot:.1f} FPS)")
    if psutil:
        print(f"RAM used by this script: {psutil.Process().memory_info().rss / 2**20:.0f} MB")
    print(f"Annotated images and results.csv saved in: {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
