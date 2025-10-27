import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
import joblib

def evaluate_models(X, y, save_prefix="models/"):
    models = {
        'GNB': GaussianNB(),
        'LR': LogisticRegression(max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=100),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    results = {}
    for name, m in models.items():
        scores = cross_val_score(m, X, y, cv=skf, scoring='accuracy')
        results[name] = {'mean_acc': scores.mean(), 'std': scores.std()}
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")
    return results

def train_and_save(X, y, model_name='RF'):
    model_map = {'GNB':GaussianNB, 'LR':LogisticRegression, 'RF':RandomForestClassifier, 'KNN':KNeighborsClassifier}
    Model = model_map[model_name]
    m = Model() if model_name!='LR' else Model(max_iter=1000)
    m.fit(X,y)
    joblib.dump(m, f"models/{model_name}.joblib")
    print("Saved", model_name)

if __name__ == "__main__":
    # baseline on MFCC
    df = pd.read_csv('data/features.csv')
    X = df[[c for c in df.columns if c.startswith('mfcc_')]].values
    y = (df['label']=='FAKE').astype(int).values
    # SMOTE balance
    sm = SMOTE(random_state=42)
    Xb, yb = sm.fit_resample(X, y)
    import os; os.makedirs('models', exist_ok=True)
    evaluate_models(Xb, yb)
    # transformed features
    tdf = pd.read_csv('data/transformed_features.csv')
    Xt = tdf[[c for c in tdf.columns if c.startswith('nmf_')]].values
    yt = (tdf['label']=='FAKE').astype(int).values
    Xt_b, yt_b = SMOTE(random_state=42).fit_resample(Xt, yt)
    evaluate_models(Xt_b, yt_b)
    # train and save an example model
    train_and_save(Xt_b, yt_b, model_name='GNB')
