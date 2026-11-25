#!/bin/bash
# PS5 Textile Defect Detection - one-time setup on Raspberry Pi 5 (Raspberry Pi OS 64-bit)
# Run from inside the ps5_pi folder:   bash setup_pi.sh
set -e

echo "== 1/4  Checking this is 64-bit Raspberry Pi OS =="
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
  echo "ERROR: this is '$ARCH', not 64-bit (aarch64). Re-flash Raspberry Pi OS (64-bit)."; exit 1
fi
python3 --version

echo "== 2/4  System packages =="
sudo apt update
sudo apt install -y python3-venv python3-pip libgl1 libglib2.0-0

echo "== 3/4  Python environment (~/ps5env) =="
python3 -m venv --system-site-packages ~/ps5env
source ~/ps5env/bin/activate
pip install --upgrade pip

echo "== 4/4  ExecuTorch runtime + Ultralytics (this can take 10-20 minutes) =="
# Same ExecuTorch version that Colab used to export the models
pip install executorch==1.6.0 ultralytics psutil

python -c "import executorch, ultralytics; print('ExecuTorch and Ultralytics installed OK')"
pip cache purge || true
sudo apt clean
df -h / | tail -1
echo
echo "Setup finished. Every time you open a new terminal, first run:"
echo "    source ~/ps5env/bin/activate"
