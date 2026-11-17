# Textile Fabric Defect Detection – PS5 (Bharat AI-SoC Challenge 2026–27)

Offline, real-time textile defect inspector on a Raspberry Pi 5.
Camera → OpenCV → YOLO26n detector → defect classifier → severity rules →
Llama 3.2 1B (ExecuTorch) advice → Streamlit dashboard + SQLite log.

## Team
- Member A – Vision (dataset, training)
- Member B – Edge / Llama (ExecuTorch, Raspberry Pi)
- Member C – App / Docs (Streamlit, severity, logging, report)
- Faculty Mentor –

## Folders
| Folder | Contents |
|---|---|
| `training/` | Training and export scripts / Colab notebooks |
| `llm/` | Llama 3.2 1B export to ExecuTorch, prompts, runner wrapper |
| `app/` | Streamlit app, camera, severity rules, SQLite logging |
| `models/` | Exported `.pte` models (or a download script if too large) |
| `scripts/` | Pi setup, benchmarks |
| `docs/` | Technical report, architecture diagram, demo video link |
| `data/` | `data.yaml` and a few sample images only (not the full dataset) |

## Setup
_To be filled in._

## Results
_To be filled in (mAP, FPS, Llama tokens/sec, RAM)._

## Dataset credits
- ZJU-Leaper fabric defect benchmark (CC BY-NC-SA 4.0) – http://www.qaas.zju.edu.cn/zju-leaper/
- https://huggingface.co/datasets/mohamedemam/zju-leaper-fabric-defect-yolo
- https://huggingface.co/datasets/aazgr/textile_defect_balanced_partial_1755872848
- https://huggingface.co/datasets/aaozgur/fabric-defect-dataset-v4
