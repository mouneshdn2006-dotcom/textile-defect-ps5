# Export the team's trained models (detector_best.pt, classifier_best.pt) for the Android app.
import sys, os, shutil
from ultralytics import YOLO
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
jobs = [("detector", 512, "/home/claude/app/det_data.yaml"), ("classifier", 224, "/home/claude/textile_dataset/classification")]
for name, size, data in jobs:
    for bits in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["fp32", "int8"]):
        args = dict(format="executorch", imgsz=size)
        if bits == "int8": args.update(quantize=8, data=data)
        m = YOLO(f"{src}/{name}_best.pt")
        out = m.export(**args)
        shutil.copy(os.path.join(out, "model.pte"), f"{dst}/{name}_{bits}.pte")
        shutil.copy(os.path.join(out, "metadata.yaml"), f"{dst}/{name}_{bits}.yaml")
        print(">>>", name, bits, round(os.path.getsize(f"{dst}/{name}_{bits}.pte") / 2**20, 2), "MB")
    if name == "classifier":
        names = YOLO(f"{src}/{name}_best.pt").names
        open(f"{dst}/classes.txt", "w").write("\n".join(names[i] for i in range(len(names))) + "\n")
        print(">>> classes:", names)
