# Textile QC – final models and app settings (v5, 24 Nov 2026)

## Models
| Model | Base | Extra training data | Size INT8 / FP32 |
|---|---|---|---|
| Detector YOLO26n (1 class "defect", 512 px) | team's trained model | +1,120 defect-free patterned fabric photos (DTD) as negatives, 1,200 original images, 4 epochs | 3.0 / 9.3 MB |
| Classifier YOLO26n-cls (6 classes, 224 px) | team's trained model | +500 DTD fabric & +800 ZJU defect-free images in "good", jute oversampled ×2, 6 epochs total | 1.6 / 5.9 MB |

Classifier test accuracy (1,681 test images): 0.808 overall (original 0.818);
per class: broken yarn 78 %, good 80 %, hole 93 %, jute 58 % (was 33 %), lycra run 92 %, stain 72 %.
Detector mAP50 on 1,500 test images: 0.729 (original 0.744) – traded for far fewer false alarms.

## App decision logic
1. Detector (confidence > 0.10, NMS IoU 0.5, max 10) → classifier names each box → severity by box size → PASS / REWORK / REJECT.
2. If the detector finds nothing, the classifier checks the whole image (close-up defects).
   Minimum confidence per class, tuned on the validation split, separately for INT8 and FP32.

## Results of the INT8 app pipeline (images never used for training or tuning)
| Test set | Result |
|---|---|
| Detection test set – defect images found (400) | **93.0 %** (original app: 84.2 %) |
| Detection test set – false alarms on good images (400) | 1.5 % (original: 0.2 %) |
| Classification test – broken yarn / hole / lycra run / stain found | **99 / 99 / 91 / 90 %** (original: 51 / 12 / 57 / 17 %) |
| Classification test – jute found | 28 % (original: 12 %) – still the weakest class |
| Classification test – false alarms on good fabric (664) | **3.3 %** (original: 47 %) |
| Patterned defect-free fabric (DTD, 280) – false alarms | **5.0 %** (original: 86 %) |
| Team's plaid shirt photos | PASS (original: REJECT) |

Credit: DTD – M. Cimpoi et al., "Describing Textures in the Wild", CVPR 2014 (research use).
