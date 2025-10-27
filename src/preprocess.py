import os
import numpy as np
import pandas as pd
from .utils import list_audio_files, read_audio
import librosa
from tqdm import tqdm

def normalize_audio(y):
    # simple peak normalization
    maxv = np.max(np.abs(y))
    if maxv > 0:
        return y / maxv
    return y

def split_to_segments(y, sr, seg_len=1.0):
    seg_samples = int(seg_len * sr)
    segments = []
    for start in range(0, len(y), seg_samples):
        seg = y[start:start+seg_samples]
        if len(seg) == seg_samples:
            segments.append(seg)
    return segments

def build_balanced_csv(audio_root, out_csv='data/DATASET-balanced.csv', sr=22050):
    rows = []
    files = list_audio_files(audio_root)
    for fp in tqdm(files, desc='Reading audio'):
        try:
            y, fs = read_audio(fp, sr=sr)
        except Exception as e:
            print("Failed:", fp, e)
            continue
        y = normalize_audio(y)
        segs = split_to_segments(y, sr)
        label = 'REAL' if os.path.sep + 'REAL' + os.path.sep in fp else 'FAKE'
        # store minimal info; feature extraction script will compute MFCC later
        for i, seg in enumerate(segs):
            fname = f"{os.path.basename(fp)}_seg{i}.npy"
            np.save(os.path.join('data','segments', fname), seg)
            rows.append({'segment_file': fname, 'label': label})
    os.makedirs('data/segments', exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("Saved", out_csv)

if __name__ == "__main__":
    # usage example: python -m src.preprocess
    build_balanced_csv('data/AUDIO', out_csv='data/DATASET-balanced.csv')
