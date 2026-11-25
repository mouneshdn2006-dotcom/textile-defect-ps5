# Textile QC – Real-Time Fabric Defect Detection & Quality-Control Assistant

**Arm Bharat AI-SoC Challenge 2026–27 · Problem Statement 5 (PS5)**

**Team**

| Name | USN |
|---|---|
| Mounesh | 1AT24EC090 |
| Krishna Patil | 1AT24EC065 |
| Ashish Pal | 1AT24EC167 |

An offline, on-device inspection system for textile fabrics. A camera image goes through two small
vision models running with **ExecuTorch** on an Arm CPU. They find the defects, name their type,
grade the severity and decide **PASS / REWORK / REJECT**. **Llama 3.2 1B**, also running with
ExecuTorch on the device, then explains the likely cause and the actions for the operator.
Every inspection is logged. Nothing needs the internet.

## What is in this repository
| Folder | Contents |
|---|---|
| `app/android/` | Android app (Kotlin, ExecuTorch 1.5.1 AAR): camera, gallery, live mode, Llama advice, log, dashboard |
| `app/raspberry_pi/` | Python version for Raspberry Pi 5 / any PC (`final.py`, `llama_advice.py`, `setup_pi.sh`) |
| `models/` | Final trained models: PyTorch `.pt` and ExecuTorch `.pte` (INT8 + FP32), with `RESULTS.md` |
| `llm/` | Llama 3.2 advisor code (ExecuTorch `TextLLMRunner`) |
| `training/` | Google Colab training notebook and the fine-tuning / evaluation scripts |
| `docs/` | Report material, screenshots, roadmap |

The Android APK is attached to the **Releases** page (on the right of the repository page).

## How it works
1. **Detector**, YOLO26n with 1 class "defect" at 512 px: finds where the defects are.
2. **Classifier**, YOLO26n-cls with 6 classes at 224 px: names each defect. The classes are hole/burst, oil stain,
   broken yarn/needle, lycra run, jute contamination and good. If the detector finds nothing, the classifier also checks the
   whole image, which catches close-up shots of a defect.
3. **Severity**: the defect's area as a share of the image (<0.5 % Low, 0.5–2 % Medium, >2 % High). Hole and broken yarn
   go up one level. **Decision**: any High → REJECT, any Medium → REWORK, otherwise PASS.
4. **Llama 3.2 1B Instruct**, exported to ExecuTorch with XNNPACK and 8da4w quantisation (914 MB): gives the likely cause and 3 actions.
5. **Log**: an `inspections.csv` row per defect (time, decision, type, severity, confidences, advice, model, ms).

## Results (images never used for training or tuning, INT8 app pipeline)
| Test | Result |
|---|---|
| Detection test set: defect images found | 93.0 % (false alarms on good images 1.5 %) |
| Classification test set: broken yarn / hole / lycra run / stain found | 99 / 99 / 91 / 90 % |
| Classification test set: jute found | 28 % (weakest class, needs more jute data) |
| Good fabric wrongly flagged (classification test / patterned fabric) | 3.3 % / 5.0 % |
| Classifier top-1 accuracy (6 classes) | 80.8 % |
| Model size INT8 (detector + classifier) | 3.0 + 1.6 MB (FP32: 9.3 + 5.9 MB) |
| Speed on phone (OnePlus Nord CE4, Snapdragon 7 Gen 3, INT8) | ~74 ms per image (no defect) |
| Llama 3.2 1B on phone | loads in ~4–9 s, an answer in ~5–40 s, fully offline |

See `models/RESULTS.md` for before/after tables and how the models were improved.

## Run it
**Android:** install the APK from Releases. Then, one time only, copy `model.pte` and `tokenizer.json` of
Llama to the phone and tap **Setup** in the app.
**Raspberry Pi 5:** see `app/raspberry_pi/HOW_TO_RUN_ON_PI.md` (`bash setup_pi.sh`, then `python final.py`).
**Llama model:** not included (914 MB, Llama 3.2 Community License). Export it yourself with the cells in the
training notebook (`optimum-cli export executorch … --qlinear 8da4w --qembedding 8w --max_seq_len 512`).

## Datasets and credits
- Detection: ZJU-Leaper fabric defect dataset (YOLO version on Hugging Face `mohamedemam/zju-leaper-fabric-defect-yolo`), CC BY-NC-SA 4.0.
- Classification: Hugging Face `aazgr/textile_defect_balanced_partial_1755872848` and `aaozgur/fabric-defect-dataset-v4`
  (no licence stated, so the images are not redistributed here).
- Defect-free patterned fabric: DTD, M. Cimpoi et al., "Describing Textures in the Wild", CVPR 2014.
- Ultralytics YOLO (AGPL-3.0), PyTorch ExecuTorch (BSD), Meta Llama 3.2 (Llama 3.2 Community License): **Built with Llama**.
