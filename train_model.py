# train_model.py
# Train a simple, robust pipeline on diabetes_user.csv and save it safely as model.joblib

import os, sys, traceback
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA = "diabetes_user.csv"
OUT = "model.joblib"
TMP = OUT + ".tmp"

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]

def load_data():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"{DATA} not found in {os.getcwd()}")
    df = pd.read_csv(DATA)
    expected = FEATURES + ["Outcome"]
    if list(df.columns) != expected:
        raise ValueError(f"CSV columns mismatch. Expected: {expected}. Found: {list(df.columns)}")
    # replace medically impossible zeros with NaN
    for c in ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]:
        df[c] = df[c].replace(0, np.nan)
    X = df[FEATURES]
    y = df["Outcome"].astype(int)
    return X, y

def build_and_train(X, y):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"Training complete. Validation accuracy: {acc:.3f}")
    return pipe

def safe_save(model):
    # atomic save: write to tmp and replace
    joblib.dump(model, TMP)
    os.replace(TMP, OUT)
    size = os.path.getsize(OUT)
    print(f"Saved model to {OUT} ({size} bytes)")

def main():
    try:
        X, y = load_data()
    except Exception as e:
        print("Data load error:", e)
        traceback.print_exc()
        sys.exit(1)
    try:
        model = build_and_train(X, y)
    except Exception as e:
        print("Training error:", e)
        traceback.print_exc()
        sys.exit(1)
    try:
        safe_save(model)
    except Exception as e:
        print("Save error:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
