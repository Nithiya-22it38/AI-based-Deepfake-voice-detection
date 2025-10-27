# AI-Powered Deepfake Voice Detection

Project to detect deepfake voices using MFCC → GNB → NMF → classical ML.

## Quick start

1. Create venv and activate:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Preprocess & extract features:

python -m src.preprocess
python -m src.features
python -m src.transfer
python -m src.train


3. Inference:

python -m src.infer --file data/AUDIO/TEST/test1.wav