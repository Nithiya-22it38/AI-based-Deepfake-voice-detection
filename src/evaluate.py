import joblib
import pandas as pd
from sklearn.metrics import classification_report

def evaluate_model(model_path, test_features_csv='data/transformed_features.csv'):
    model = joblib.load(model_path)
    df = pd.read_csv(test_features_csv)
    X = df[[c for c in df.columns if c.startswith('nmf_')]].values
    y = (df['label']=='FAKE').astype(int).values
    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == "__main__":
    evaluate_model('models/GNB.joblib')
