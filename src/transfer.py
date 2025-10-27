# src/transfer.py
import os
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import NMF
from sklearn.preprocessing import LabelEncoder
import joblib

def gnb_nmf_transform(features_csv='data/features.csv', out_csv='data/transformed_features.csv',
                      nmf_components=20, save_transformers=True):
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv(features_csv)
    X = df[[c for c in df.columns if c.startswith('mfcc_')]].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # 1) GNB used to produce probabilistic features
    gnb = GaussianNB()
    gnb.fit(X, y)
    prob = gnb.predict_proba(X)  # shape (n_samples, n_classes)

    # 2) combine and ensure non-negativity
    combined = np.hstack([X, prob])
    minv = combined.min()
    if minv < 0:
        combined = combined - minv

    # 3) NMF
    nmf = NMF(n_components=nmf_components, init='random', random_state=42, max_iter=500)
    W = nmf.fit_transform(combined)  # transformed features

    # 4) save transformed features
    out_df = pd.DataFrame(W, columns=[f'nmf_{i}' for i in range(W.shape[1])])
    out_df['label'] = df['label'].values
    out_df.to_csv(out_csv, index=False)
    print("Saved transformed features to", out_csv)

    # 5) Save the gnb and nmf objects for use at inference time
    if save_transformers:
        joblib.dump(gnb, os.path.join('models', 'transform_gnb.joblib'))
        joblib.dump(nmf, os.path.join('models', 'transform_nmf.joblib'))
        joblib.dump(le, os.path.join('models', 'label_encoder.joblib'))  # optional
        print("Saved transformer objects to models/")

    return out_df

if __name__ == "__main__":
    gnb_nmf_transform()
