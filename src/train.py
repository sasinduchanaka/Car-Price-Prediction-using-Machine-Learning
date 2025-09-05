
import re, json, joblib, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/Car details v3.csv")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(exist_ok=True, parents=True)

# --- helpers ---
def std_cols(cols):
    return (cols.str.strip().str.lower().str.replace(" ", "_").str.replace("-", "_"))

def extract_first_number(x):
    if pd.isna(x): return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(x))
    return float(m.group()) if m else np.nan

def torque_to_nm(x):
    if pd.isna(x): return np.nan
    s = str(x).lower().replace(" ", "")
    m_nm = re.search(r"(\d*\.?\d+)n?m", s)
    if m_nm: return float(m_nm.group(1))
    m_kgm = re.search(r"(\d*\.?\d+)kgm", s)
    if m_kgm: return float(m_kgm.group(1))*9.80665
    m_any = re.search(r"(\d*\.?\d+)", s)
    return float(m_any.group(1)) if m_any else np.nan

def clean_dataframe(df):
    df = df.copy()
    df.columns = std_cols(df.columns)
    df = df.drop_duplicates()

    for col in ["mileage","engine","max_power"]:
        if col in df.columns:
            df[col] = df[col].apply(extract_first_number)

    if "torque" in df.columns:
        df["torque_nm"] = df["torque"].apply(torque_to_nm)

    for col in ["seats","year","km_driven","selling_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["selling_price"] > 0]
    if "year" in df.columns:
        df = df[df["year"].between(1985,2025)]
    if "km_driven" in df.columns:
        df = df[df["km_driven"].between(0,1_500_000)]

    if "year" in df.columns:
        df["car_age"] = 2025 - df["year"]
    if "name" in df.columns:
        df["brand"] = df["name"].astype(str).str.split().str[0]
        df = df.drop(columns=["name"])

    return df

def build_preprocessor(df):
    numeric_cols = [c for c in ["year","car_age","km_driven","mileage","engine","max_power","seats","torque_nm"] if c in df.columns]
    categorical_cols = [c for c in ["fuel","seller_type","transmission","brand"] if c in df.columns]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first"))])

    preprocessor = ColumnTransformer(transformers=[("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
    meta = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "target_col": "selling_price"}
    return preprocessor, meta

def evaluate(pipe, X_train, X_valid, y_train, y_valid):
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    mse = mean_squared_error(y_valid, preds)
    return {
    "r2": float(r2_score(y_valid, preds)),
    "mae": float(mean_absolute_error(y_valid, preds)),
    "rmse": float(mse ** 0.5)
}


# --- main ---
if __name__ == "__main__":
    df_raw = pd.read_csv(DATA_PATH)
    df = clean_dataframe(df_raw)
    assert "selling_price" in df.columns

    y = df["selling_price"].copy()
    X = df.drop(columns=["selling_price"]).copy()

    preprocessor, meta = build_preprocessor(df)

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    pipeline = Pipeline([("pre", preprocessor), ("model", rf)])

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    base_metrics = evaluate(pipeline, X_train, X_valid, y_train, y_valid)
    print("Base metrics:", base_metrics)

    # Randomized search
    param_grid = {
        "model__n_estimators":[200,400,600],
        "model__max_depth":[None,10,18,26],
        "model__min_samples_split":[2,5,10],
        "model__min_samples_leaf":[1,2,4],
        "model__max_features":["sqrt","log2",0.5]
    }

    search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_iter=20, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1, random_state=42, verbose=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    best_metrics = evaluate(best, X_train, X_valid, y_train, y_valid)

    # save artifacts
    joblib.dump(best, ART_DIR / "best_model.joblib")
    joblib.dump(meta, ART_DIR / "inference_schema.joblib")
    with open(ART_DIR / "metrics.json","w") as f: json.dump({"base":base_metrics,"best":best_metrics,"best_params":search.best_params_}, f, indent=2)

    print("Saved best_model.joblib and inference_schema.joblib to artifacts/")
    print("Metrics:", best_metrics)
