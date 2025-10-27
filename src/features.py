import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

def mfcc_from_segment(seg_path, sr=22050, n_mfcc=13):
    y = np.load(seg_path)
    mfcc = librosa.feature.mfcc(y=y.astype(float), sr=sr, n_mfcc=n_mfcc)
    # aggregate over time (mean + std)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def build_features(csv_in='data/DATASET-balanced.csv', out_features='data/features.csv', sr=22050):
    df = pd.read_csv(csv_in)
    feats = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        seg_file = os.path.join('data','segments', row['segment_file'])
        if not os.path.exists(seg_file):
            continue
        f = mfcc_from_segment(seg_file, sr=sr, n_mfcc=20)
        feats.append({'segment_file': row['segment_file'], 'label': row['label'], **{f"mfcc_{i}": float(v) for i,v in enumerate(f)}})
    pf = pd.DataFrame(feats)
    pf.to_csv(out_features, index=False)
    print("Saved", out_features)

if __name__ == "__main__":
    build_features()
