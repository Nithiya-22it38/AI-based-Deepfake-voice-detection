# src/infer.py
import os
import argparse
import numpy as np
import joblib
import soundfile as sf
import librosa
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder

# helper MFCC (must match features.py)
def mfcc_from_audio_array(y, sr=22050, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y.astype(float), sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def split_to_segments(y, sr, seg_len=1.0):
    seg_samples = int(seg_len * sr)
    segments = []
    for start in range(0, len(y), seg_samples):
        seg = y[start:start+seg_samples]
        if len(seg) == seg_samples:
            segments.append(seg)
    return segments

def ensure_transfer_objects(features_csv='data/features.csv', gnb_path='models/transfer_gnb.joblib', nmf_path='models/transfer_nmf.joblib', nmf_components=20):
    """Load transfer objects if saved; otherwise fit on training features and save them."""
    if os.path.exists(gnb_path) and os.path.exists(nmf_path):
        gnb = joblib.load(gnb_path)
        nmf = joblib.load(nmf_path)
        print("Loaded existing transfer objects.")
        return gnb, nmf

    # need to fit them from training features
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Training features file not found: {features_csv}")

    import pandas as pd
    df = pd.read_csv(features_csv)
    X = df[[c for c in df.columns if c.startswith('mfcc_')]].values
    labels = df['label'].values
    le = LabelEncoder()
    y = le.fit_transform(labels)  # 0/1

    # fit gnb on training MFCCs
    gnb = GaussianNB()
    gnb.fit(X, y)
    prob = gnb.predict_proba(X)  # shape (n_samples, n_classes)

    combined = np.hstack([X, prob])
    minv = combined.min()
    if minv < 0:
        combined = combined - minv

    nmf = NMF(n_components=nmf_components, init='random', random_state=42, max_iter=500)
    nmf.fit(combined)

    os.makedirs(os.path.dirname(gnb_path), exist_ok=True)
    joblib.dump(gnb, gnb_path)
    joblib.dump(nmf, nmf_path)
    print(f"Fitted and saved transfer objects to {gnb_path} and {nmf_path}")
    return gnb, nmf

def transform_single_mfcc(mfcc_vec, gnb, nmf):
    # mfcc_vec shape (d,) -> make 2D
    X = mfcc_vec.reshape(1, -1)
    prob = gnb.predict_proba(X)  # (1, classes)
    combined = np.hstack([X, prob])
    minv = combined.min()
    if minv < 0:
        combined = combined - minv
    W = nmf.transform(combined)  # (1, nmf_components)
    return W.ravel()

def load_classifier(model_path='models/GNB.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained classifier not found: {model_path}")
    clf = joblib.load(model_path)
    return clf

def predict_file(file_path, model_path='models/GNB.joblib', features_csv='data/features.csv', sr=22050, n_mfcc=20):
    # read audio
    y, fs = sf.read(file_path)
    if fs != sr:
        y = librosa.resample(y.astype(float), orig_sr=fs, target_sr=sr)
    # split to segments
    segments = split_to_segments(y, sr, seg_len=1.0)
    if len(segments) == 0:
        raise ValueError("Audio shorter than 1 second or no full 1-second segments found.")
    # ensure transfer objects
    gnb_tr, nmf = ensure_transfer_objects(features_csv=features_csv)
    clf = load_classifier(model_path)

    per_seg_preds = []
    per_seg_probs = []
    for seg in segments:
        mf = mfcc_from_audio_array(seg, sr=sr, n_mfcc=n_mfcc)
        nmf_feat = transform_single_mfcc(mf, gnb_tr, nmf).reshape(1, -1)
        # for classifiers that output predict_proba:
        if hasattr(clf, "predict_proba"):
            prob = clf.predict_proba(nmf_feat)[0]  # [prob_class0, prob_class1]
            pred = clf.predict(nmf_feat)[0]
            # assume class 1 = FAKE (this depends on how you encoded training labels)
            per_seg_preds.append(int(pred))
            per_seg_probs.append(prob)
        else:
            pred = clf.predict(nmf_feat)[0]
            per_seg_preds.append(int(pred))
            per_seg_probs.append(None)

    # aggregate:
    # majority vote
    votes = np.array(per_seg_preds)
    avg_prob = None
    if per_seg_probs and per_seg_probs[0] is not None:
        # take average of probability of class 1 (FAKE)
        probs_class1 = np.array([p[1] for p in per_seg_probs])
        avg_prob = float(probs_class1.mean())

    majority = int(np.round(votes.mean()))  # 0 or 1
    label_map = {0: "REAL", 1: "FAKE"}
    return {
        "file": file_path,
        "segments": len(segments),
        "per_segment_predictions": per_seg_preds,
        "per_segment_probabilities": per_seg_probs,
        "aggregated_vote": label_map[majority],
        "aggregated_fake_probability": avg_prob
    }

def main():
    parser = argparse.ArgumentParser(description="Infer REAL/FAKE for an audio file")
    parser.add_argument("--file", "-f", required=True, help="Path to audio file (.wav, .mp3)")
    parser.add_argument("--model", "-m", default="models/GNB.joblib", help="Saved classifier")
    parser.add_argument("--features", default="data/features.csv", help="Training features CSV (used to fit transfer objects if needed)")
    args = parser.parse_args()

    out = predict_file(args.file, model_path=args.model, features_csv=args.features)
    print("File:", out['file'])
    print("Segments analysed:", out['segments'])
    print("Aggregated prediction (majority):", out['aggregated_vote'])
    if out['aggregated_fake_probability'] is not None:
        print(f"Avg FAKE probability across segments: {out['aggregated_fake_probability']:.3f}")
    print("Per-segment predictions (0=REAL, 1=FAKE):", out['per_segment_predictions'])

if __name__ == "__main__":
    main()
