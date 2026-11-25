# Run the trained models on the Raspberry Pi (Step 11)

## What you need
- Raspberry Pi 5 with **Raspberry Pi OS (64-bit)**, connected to the internet (for setup only)
- A pen drive
- From Google Drive → `Textile fabric/models`: the model folders (`..._fp32` and `..._int8`)
- `textile_demo_test_images.zip` (64 test images)
- This `ps5_pi` folder

## 1. Put everything on the pen drive (on your laptop)
1. In Google Drive, open **Textile fabric → models**, right-click `detector_executorch` → **Download**, then the same for `classifier_executorch`. Drive gives you zips: extract them.
2. Copy to the pen drive, arranged like this:
```
ps5_pi/
  pi_test.py
  setup_pi.sh
  HOW_TO_RUN_ON_PI.md
  models/
    detector_executorch_fp32/     (32-bit, contains model.pte + metadata.yaml)
    classifier_executorch_fp32/
    detector_executorch_int8/     (8-bit, smaller and faster)
    classifier_executorch_int8/
  demo_test_images/           (the 64 images, unzipped)
```
Check: each model folder must contain **model.pte** and **metadata.yaml** directly inside it (not inside another folder).

## 2. Copy to the Pi
Plug the pen drive into the Pi, open the file manager and copy the whole `ps5_pi` folder to your **home folder** (`/home/<your-username>/ps5_pi`).

## 3. One-time setup (10–20 minutes)
Open **Terminal** on the Pi and type:
```bash
cd ~/ps5_pi
bash setup_pi.sh
```
It ends with `ExecuTorch and Ultralytics installed OK`.

## 4a. MAIN PROGRAM: textile_inspector.py (photos or live camera)
Works from **any Python editor** (Thonny on the Pi, VS Code, IDLE, PyCharm) or the terminal.
1. Open `textile_inspector.py` in Thonny (Pi menu → Programming → Thonny).
   Thonny: Tools → Options → Interpreter → choose `~/ps5env/bin/python3` (the environment from setup_pi.sh).
2. At the top, change the SETTINGS:
   - `MODE = "folder"` (all demo photos), `"image"` (one photo, set `INPUT`), or `"camera"` (live)
   - `BITS = "int8"` (8-bit) or `"fp32"` (32-bit)
3. Press **Run**. A window shows each result; press any key for the next photo, **q** to quit.
   In camera mode: **space** = log this inspection (+ Llama advice), **q** = quit.
4. Results: annotated pictures in `results/`, every inspection in `inspections.db` and `inspections.csv`.

Terminal version:
```bash
source ~/ps5env/bin/activate
python textile_inspector.py                      # uses the settings in the file
python textile_inspector.py --mode camera        # live camera
python textile_inspector.py --bits fp32          # 32-bit models
```
**On a Windows/Mac laptop:** install Python 3.11+, run `pip install -r requirements.txt`, then run the same file.

## 4. Run the speed test
```bash
cd ~/ps5_pi
source ~/ps5env/bin/activate
python pi_test.py --images demo_test_images
```
You will see one line per image, for example
```
[12/64] hole_burst__1a2b.jpg: 1 defect(s) ['hole_burst:High'] -> REJECT  (180 ms)
```
and a **SUMMARY** at the end with speed (ms per image, FPS) and RAM.
Ignore lines starting with `[cpuinfo_utils.cpp…]` – that is ExecuTorch reading the CPU type.

## 4b. Run the faster INT8 (8-bit) models
If you exported the 8-bit models in Colab, put `detector_executorch_int8` and `classifier_executorch_int8` in `models/` too, then:
```bash
python pi_test.py --images demo_test_images --det models/detector_executorch_int8 --cls models/classifier_executorch_int8
```
Run both versions and write both SUMMARY speeds in the report (32-bit vs 8-bit).

## 4c. Llama 3.2 advice (mandatory part of PS5)
1. From Google Drive → `Textile fabric/llama`, download `model.pte`, `tokenizer.json`, `tokenizer_config.json` and put them in `ps5_pi/llama/`.
2. Run:
```bash
python llama_advice.py --defect hole_burst --severity High --size 6.1
```
It loads Llama 3.2 1B (a few seconds) and prints a short "Likely cause / Actions" answer, generated fully offline.
Try `--defect stain_oil --severity Medium --decision REWORK` too. Note the time it prints – it goes in the report.

## 5. Look at the results
Open the `results` folder in the file manager:
- each image with coloured boxes: green = Low, orange = Medium, red = High, and PASS / REWORK / REJECT at the top
- `results.csv`: one row per defect (type, confidence, size, severity, decision) – opens in LibreOffice Calc

**Send a photo/screenshot of the SUMMARY** – these speed numbers go in the report.

## Useful options
| Option | Default | Meaning |
|---|---|---|
| `--conf 0.15` | 0.20 | Lower = finds more defects (higher recall) but more false alarms |
| `--max-det 5` | 10 | Max defects checked per image (fewer = faster) |
| `--images my_photos` | demo_test_images | Any folder of .jpg/.png photos, e.g. your own fabric photos |

## Severity rules used
- Size of the defect box compared with the whole image: < 0.5 % = Low, 0.5–2 % = Medium, > 2 % = High
- **Holes and broken yarn are one level more serious** (they weaken the fabric)
- Decision per image: any High → **REJECT**, any Medium → **REWORK**, otherwise **PASS**

These thresholds are a starting point – adjust them after testing on your own fabric.

## If something goes wrong
| Message | Fix |
|---|---|
| `this is 'armv7l', not 64-bit` | Re-flash the SD card with **Raspberry Pi OS (64-bit)** |
| `No images found` | Check the folder name after `--images` |
| `not a supported model format` | Each model folder must contain `model.pte` directly inside it |
| `No module named ultralytics` | Run `source ~/ps5env/bin/activate` first |
| Very slow / Pi freezes | Use the official 27 W power supply + active cooler; try `--max-det 5` |
