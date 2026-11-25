"""
PS5 - Textile Fabric Defect Inspector (runs fully offline with ExecuTorch)
=========================================================================
Works on Raspberry Pi 5, Windows, Linux and Mac (Python 3.11+), from any editor
(Thonny, VS Code, IDLE, PyCharm) or the terminal.

HOW TO USE
  1. Change the SETTINGS below (MODE, INPUT, ...).
  2. Press Run (or in a terminal:  python textile_inspector.py).
     Terminal users can also override settings, e.g.:
        python textile_inspector.py --mode camera
        python textile_inspector.py --mode image --input demo_test_images/some_photo.jpg

Needs:  pip install executorch==1.6.0 ultralytics
Files (next to this script):
  models/detector_executorch_int8/model.pte     (and _fp32 versions)
  models/classifier_executorch_int8/model.pte
  llama/model.pte + llama/tokenizer.json         (optional - Llama 3.2 advice)
  models/pytorch/detector_best.pt + classifier_best.pt   (optional - laptop fallback when
        Windows blocks ExecuTorch; same trained models in PyTorch format)
"""
# ============================== SETTINGS ==============================
MODE = "folder"            # "image" = one photo, "folder" = all photos in a folder, "camera" = live camera
INPUT = "demo_test_images" # photo file or folder (for "image" / "folder")
CAMERA = 0                 # 0 = first camera (USB webcam or Pi Camera); or a video file path
BITS = "int8"              # "int8" (8-bit, faster/smaller) or "fp32" (32-bit)
ENGINE = "auto"            # "auto" = ExecuTorch .pte, or PyTorch .pt if ExecuTorch is blocked (e.g. Windows)
                           # "executorch" = only .pte    "pytorch" = only .pt (models/pytorch/)
CONF = 0.20                # detection confidence: lower = finds more defects
USE_LLAMA = True           # add Llama 3.2 advice (only if the llama/ files exist)
SHOW_WINDOW = True         # show result pictures in a window (needs a screen)
FABRIC = "woven cotton"    # fabric type (used in the Llama advice)
# =====================================================================

import argparse, csv, glob, os, shutil, sqlite3, sys, time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))          # works from any editor / working folder
def here(*p): return os.path.join(HERE, *p)

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"Missing package: {e.name}\nInstall with:  pip install executorch==1.6.0 ultralytics")

CRITICAL = {"hole_burst", "broken_yarn_needle"}
COLORS = {"Low": (0, 200, 0), "Medium": (0, 165, 255), "High": (0, 0, 255)}
DEC_COLORS = {"PASS": (0, 160, 0), "REWORK": (0, 165, 255), "REJECT": (0, 0, 255)}


# ---------------------------------------------------------------- models
def executorch_ok():
    """True if the ExecuTorch runtime can load on this computer."""
    try:
        from executorch.runtime import Runtime  # noqa: F401
        return True
    except Exception as e:  # e.g. Windows Smart App Control blocks its DLL
        print(f"ExecuTorch runtime not available here: {str(e).splitlines()[0]}")
        return False


def load_pytorch(kind, task):
    pt = here("models", "pytorch", f"{kind}_best.pt")
    if not os.path.isfile(pt):
        sys.exit(f"PyTorch model not found: {pt}\nDownload best.pt from Google Drive -> Textile fabric -> runs -> "
                 f"{kind} -> weights, rename it to {kind}_best.pt and put it in models/pytorch/.")
    return YOLO(pt, task=task)


def load_models():
    """Returns (detector, classifier, engine name)."""
    global ENGINE
    if ENGINE == "pytorch" or (ENGINE == "auto" and not executorch_ok()):
        if ENGINE == "auto":
            print("Using the PyTorch .pt models instead (same trained models). On the Raspberry Pi the .pte models are used.")
        ENGINE = "pytorch"
        return load_pytorch("detector", "detect"), load_pytorch("classifier", "classify"), "PyTorch .pt"
    ENGINE = "executorch"
    return load_model("detector", BITS, "detect"), load_model("classifier", BITS, "classify"), f"ExecuTorch .pte {BITS}"


def load_model(kind, bits, task):
    """Load models/<kind>_executorch_<bits>. Ultralytics needs the folder name to end in
    '_executorch_model', so a small copy with that name is made once (works on every OS)."""
    src = here("models", f"{kind}_executorch_{bits}")
    if not os.path.isfile(os.path.join(src, "model.pte")):
        sys.exit(f"Model not found: {os.path.join(src, 'model.pte')}\nPut the model folders from Google Drive into the 'models' folder.")
    alias = here("models", f".{kind}_{bits}_executorch_model")
    if not os.path.isdir(alias) or os.path.getmtime(os.path.join(alias, "model.pte")) < os.path.getmtime(os.path.join(src, "model.pte")):
        shutil.rmtree(alias, ignore_errors=True)
        shutil.copytree(src, alias)
    return YOLO(alias, task=task)


def load_llama():
    if not USE_LLAMA:
        return None
    m, t = here("llama", "model.pte"), here("llama", "tokenizer.json")
    if not (os.path.isfile(m) and os.path.isfile(t)):
        print("Llama advice: OFF (put model.pte + tokenizer.json in the 'llama' folder to turn it on)")
        return None
    try:
        sys.path.insert(0, HERE)
        from llama_advice import LlamaAdvisor
        t0 = time.time()
        adv = LlamaAdvisor(m, t)
        print(f"Llama 3.2 loaded in {time.time() - t0:.1f} s")
        return adv
    except Exception as e:
        print("Llama advice: OFF -", e)
        return None


# ---------------------------------------------------------------- logic
def severity(dtype, area_pct):
    lvl = 0 if area_pct < 0.5 else (1 if area_pct < 2.0 else 2)
    if dtype in CRITICAL:
        lvl = min(lvl + 1, 2)
    return ["Low", "Medium", "High"][lvl]


def decide(sevs):
    return "REJECT" if "High" in sevs else ("REWORK" if "Medium" in sevs else "PASS")


def inspect(img, det, cls):
    """Return (annotated image, decision, list of defects, ms)."""
    t0 = time.perf_counter()
    H, W = img.shape[:2]
    out = img.copy()
    r = det.predict(img, imgsz=512, conf=CONF, max_det=10, verbose=False)[0]
    fs = max(0.45, min(W, H) / 900)
    th = 1 if fs < 0.6 else 2
    defects = []
    for (x0, y0, x1, y1), score in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        bw, bh = x1 - x0, y1 - y0
        crop = img[int(max(0, y0 - .15 * bh)):int(min(H, y1 + .15 * bh)), int(max(0, x0 - .15 * bw)):int(min(W, x1 + .15 * bw))]
        if crop.size == 0:
            continue
        c = cls.predict(crop, imgsz=224, verbose=False)[0]
        dtype = c.names[c.probs.top1]
        if dtype == "good":
            dtype = "unclassified_defect"
        area = 100.0 * bw * bh / (W * H)
        sev = severity(dtype, area)
        defects.append({"type": dtype, "severity": sev, "det_conf": round(score, 3),
                        "type_conf": round(float(c.probs.top1conf), 3), "area_pct": round(area, 2)})
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), COLORS[sev], th + 1)
        cv2.putText(out, f"{dtype} {sev}", (int(x0), max(15, int(y0) - 5)), cv2.FONT_HERSHEY_SIMPLEX, fs, COLORS[sev], th, cv2.LINE_AA)
    dec = decide([d["severity"] for d in defects])
    cv2.putText(out, dec, (10, int(32 * fs * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.6, DEC_COLORS[dec], th + 1, cv2.LINE_AA)
    return out, dec, defects, (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------- logging
def open_log():
    db = sqlite3.connect(here("inspections.db"))
    db.execute("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, source TEXT, decision TEXT,
        defect_type TEXT, severity TEXT, det_conf REAL, type_conf REAL, area_pct REAL,
        advice TEXT, model_bits TEXT, ms REAL)""")
    return db


def log(db, source, dec, defects, advice, ms):
    now = datetime.now().isoformat(timespec="seconds")
    rows = defects or [{"type": "none", "severity": "", "det_conf": None, "type_conf": None, "area_pct": 0}]
    for d in rows:
        db.execute("INSERT INTO inspections (time, source, decision, defect_type, severity, det_conf, type_conf, area_pct, advice, model_bits, ms) "
                   "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (now, source, dec, d["type"], d["severity"], d["det_conf"], d["type_conf"], d["area_pct"], advice,
                    "pytorch" if ENGINE == "pytorch" else BITS, round(ms, 1)))
    db.commit()


def export_csv(db):
    path = here("inspections.csv")
    cur = db.execute("SELECT * FROM inspections ORDER BY id")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([c[0] for c in cur.description])
        w.writerows(cur.fetchall())
    return path


# ---------------------------------------------------------------- output helpers
def get_advice(llama, dec, defects):
    if not llama or not defects:
        return ""
    worst = max(defects, key=lambda d: ["Low", "Medium", "High"].index(d["severity"]))
    text, secs = llama.advise(worst["type"], worst["severity"], worst["area_pct"], dec)
    print(f"  Llama advice ({secs:.1f} s):\n    " + text.replace("\n", "\n    "))
    return text


def report(name, dec, defects, ms):
    print(f"\n=== {name} -> {dec}  ({ms:.0f} ms) ===")
    if not defects:
        print("  No defect found")
    for d in defects:
        print(f"  {d['type']:22s} {d['severity']:6s} | found {d['det_conf']:.2f} | type {d['type_conf']:.2f} | size {d['area_pct']:.1f}%")


def screen_available():
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def show(title, img, wait_ms):
    global SHOW_WINDOW
    if SHOW_WINDOW and not screen_available():
        print("(No screen detected - window display turned off; results are still saved.)")
        SHOW_WINDOW = False
    if not SHOW_WINDOW:
        return -1
    try:
        cv2.imshow(title, img)
        return cv2.waitKey(wait_ms) & 0xFF
    except cv2.error:
        print("(No screen available - window display turned off; results are still saved.)")
        SHOW_WINDOW = False
        return -1


# ---------------------------------------------------------------- modes
def run_files(paths, det, cls, llama, db):
    os.makedirs(here("results"), exist_ok=True)
    times = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("  skip (cannot read):", p)
            continue
        out, dec, defects, ms = inspect(img, det, cls)
        times.append(ms)
        report(os.path.basename(p), dec, defects, ms)
        advice = get_advice(llama, dec, defects)
        log(db, os.path.basename(p), dec, defects, advice, ms)
        cv2.imwrite(here("results", os.path.basename(p)), out)
        key = show("Textile Inspector - press any key for next, q to quit", out, 0 if len(paths) < 20 else 600)
        if key == ord("q"):
            break
    if times:
        print(f"\nAverage {sum(times) / len(times):.0f} ms per image ({1000 * len(times) / sum(times):.1f} FPS), {len(times)} images, "
              f"{'PyTorch .pt' if ENGINE == 'pytorch' else 'ExecuTorch ' + BITS} models")
        print("Annotated pictures saved in:", here("results"))


def run_camera(det, cls, llama, db):
    picam = None
    try:  # Raspberry Pi Camera Module
        from picamera2 import Picamera2
        picam = Picamera2()
        picam.configure(picam.create_preview_configuration(main={"size": (1280, 960), "format": "RGB888"}))
        picam.start()
        grab = lambda: (True, picam.capture_array())
        print("Using Raspberry Pi Camera")
    except Exception:
        cap = cv2.VideoCapture(CAMERA)
        if not cap.isOpened():
            sys.exit(f"Cannot open camera {CAMERA}. Check the cable, or set CAMERA to a video file.")
        grab = cap.read
        print(f"Using camera / video: {CAMERA}")
    print("Live inspection running.  Keys:  [space] = log this inspection (+ Llama advice)   [q] = quit")
    os.makedirs(here("results"), exist_ok=True)
    n = 0
    last_logged = None
    while True:
        ok, frame = grab()
        if not ok or frame is None:
            break
        out, dec, defects, ms = inspect(frame, det, cls)
        cv2.putText(out, f"{1000 / max(ms, 1):.1f} FPS ({BITS})", (10, out.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        n += 1
        key = show("Textile Inspector - LIVE  (space = log, q = quit)", out, 1)
        if not SHOW_WINDOW:  # no screen: log automatically when the decision changes
            key = ord(" ") if dec != last_logged else -1
        if key == ord(" "):
            name = datetime.now().strftime("cam_%Y%m%d_%H%M%S.jpg")
            report(name, dec, defects, ms)
            advice = get_advice(llama, dec, defects)
            log(db, name, dec, defects, advice, ms)
            cv2.imwrite(here("results", name), out)
            last_logged = dec
        if key == ord("q"):
            break
    if picam:
        picam.stop()


# ---------------------------------------------------------------- main
def main():
    global MODE, INPUT, BITS, CONF, USE_LLAMA, SHOW_WINDOW, CAMERA, ENGINE
    ap = argparse.ArgumentParser(description="PS5 textile defect inspector")
    ap.add_argument("--mode", choices=["image", "folder", "camera"])
    ap.add_argument("--input")
    ap.add_argument("--camera")
    ap.add_argument("--bits", choices=["int8", "fp32"])
    ap.add_argument("--engine", choices=["auto", "executorch", "pytorch"])
    ap.add_argument("--conf", type=float)
    ap.add_argument("--no-llama", action="store_true")
    ap.add_argument("--no-window", action="store_true")
    a, _ = ap.parse_known_args()  # unknown args from IDEs are ignored
    MODE = a.mode or MODE
    INPUT = a.input or INPUT
    BITS = a.bits or BITS
    ENGINE = a.engine or ENGINE
    CONF = a.conf if a.conf is not None else CONF
    USE_LLAMA = USE_LLAMA and not a.no_llama
    SHOW_WINDOW = SHOW_WINDOW and not a.no_window
    if a.camera is not None:
        CAMERA = int(a.camera) if a.camera.isdigit() else a.camera

    print(f"PS5 Textile Inspector | mode={MODE} | models={BITS} | conf={CONF}")
    det, cls, engine = load_models()
    print(f"Models loaded ({engine})")
    llama = load_llama()
    db = open_log()

    if MODE == "camera":
        run_camera(det, cls, llama, db)
    else:
        src = INPUT if os.path.isabs(INPUT) else here(INPUT)
        if MODE == "image":
            paths = [src]
        else:
            paths = sorted(p for ext in ("jpg", "jpeg", "png", "bmp") for p in glob.glob(os.path.join(src, f"*.{ext}")))
        if not paths or not os.path.exists(paths[0]):
            sys.exit(f"No photos found at: {src}")
        run_files(paths, det, cls, llama, db)

    total = db.execute("SELECT COUNT(DISTINCT time || source) FROM inspections").fetchone()[0]
    print(f"\nLog: {here('inspections.db')}  ({total} inspections so far)")
    print("CSV export:", export_csv(db))
    db.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
