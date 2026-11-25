# Textile Fabric Defect Dataset – PS5 (Bharat AI-SoC Challenge 2026–27)

Collected and prepared on 16 Nov 2026 for Phase 1.
Two parts, matching the two-stage design: **detection** finds *where* the defect is,
**classification** says *what type* it is.

## 1. `detection/` – YOLO format, 1 class (`defect`)

| Split | Images | With defects | Defect-free | Boxes |
|---|---:|---:|---:|---:|
| train | 12,412 | 5,476 | 6,936 | 7,913 |
| val   | 2,190  | 948   | 1,242 | 1,369 |
| test  | 3,360  | 2,256 | 1,104 | 3,436 |

- 512×512 colour images of real fabrics (plain, striped, checked, printed).
- Defect-free images have an empty label file (normal YOLO convention; they reduce false alarms).
- `test` = the official ZJU-Leaper test split, so it never overlaps with train.
- Source: ZJU-Leaper (YOLO version by mohamedemam on Hugging Face), boxes made from the
  original segmentation masks. Licence **CC BY-NC-SA 4.0** (non-commercial, credit, share-alike).

## 2. `classification/` – folder-per-class, 6 classes

| Class folder | Meaning | train | val | test |
|---|---|---:|---:|---:|
| `hole_burst` | Hole / burst | 1,041 | 223 | 222 |
| `stain_oil` | Oil stain | 435 | 94 | 92 |
| `broken_yarn_needle` | Broken yarn / needle break | 367 | 79 | 77 |
| `lycra_run` | Lycra run (broken lycra thread line) | 1,763 | 378 | 377 |
| `jute` | Jute contamination class from the source set | 1,168 | 251 | 249 |
| `good` | No defect | 3,105 | 666 | 664 |

- Split 70 / 15 / 15. Exact duplicates removed (140) and images under 96 px dropped (114).
- Near-duplicates (same image rotated/flipped/resized) were grouped and kept in the
  **same** split so test accuracy is not inflated. Small-angle rotated copies may still slip through.
- Images resized so the longest side is at most 640 px.
- Sources: `aazgr/textile_defect_balanced_partial_1755872848` and `aaozgur/fabric-defect-dataset-v4`
  on Hugging Face. **No licence is stated** on these – use for the competition, credit them,
  and do not re-publish the images in your public GitHub repo (link to them instead).

## 3. `demo_test_images/` – 64 held-out images
48 classification test images (8 per class, name starts with the class) + 16 detection test
images with defects (`detect__...`). Use these to try the app on the Raspberry Pi.

## Training (Google Colab, GPU)

```bash
pip install ultralytics
# Stage 1 – detector (where is the defect?)
yolo detect train model=yolo26n.pt data=detection/data.yaml imgsz=512 epochs=60 batch=32
# Stage 2 – type classifier (what is it?)
yolo classify train model=yolo26n-cls.pt data=classification imgsz=224 epochs=40 batch=64
# Export both for the Pi with ExecuTorch
yolo export model=runs/detect/train/weights/best.pt format=executorch
yolo export model=runs/classify/train/weights/best.pt format=executorch
```

On the Pi: run the detector on each frame, crop each box, run the classifier on the crop,
then pass type + severity to Llama (ExecuTorch) for advice.

## Still needed: your own images
Public data will not match your camera and lighting. Before the demo, add 150–300 photos of
your chosen fabric under the LED box (both good and defective), label them, and add them to
train/val. This usually gives the biggest accuracy jump.

## Credits
- ZJU-Leaper fabric defect benchmark (Zhejiang University) – http://www.qaas.zju.edu.cn/zju-leaper/
- https://huggingface.co/datasets/mohamedemam/zju-leaper-fabric-defect-yolo
- https://huggingface.co/datasets/aazgr/textile_defect_balanced_partial_1755872848
- https://huggingface.co/datasets/aaozgur/fabric-defect-dataset-v4
