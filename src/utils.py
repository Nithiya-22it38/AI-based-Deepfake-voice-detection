import os
import numpy as np
import soundfile as sf

def list_audio_files(root_dir):
    files = []
    for root, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.wav','.flac','.mp3','.ogg')):
                files.append(os.path.join(root, f))
    return files

def read_audio(path, sr=None):
    # returns (y, sr)
    y, fs = sf.read(path)
    if sr is not None and fs != sr:
        # keep simple: librosa can resample if needed elsewhere
        import librosa
        y = librosa.resample(y.astype(float), orig_sr=fs, target_sr=sr)
        fs = sr
    return y, fs
