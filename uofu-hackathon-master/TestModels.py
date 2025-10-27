#!/usr/bin/env python3
"""
Evaluation-only script for Mars temperature models.

- Loads ./models/model_max_temp.joblib and ./models/model_min_temp.joblib
- Rebuilds features from a CSV next to this script
- Uses the SAME 85/15 time-safe split as training
- Applies any saved bias model (isotonic) and quantile scale from the bundles
- Prints comparable metrics:
    * MAE, R^2, sMAPE (reference)
    * Accuracy_IQR = 100 * (1 - MAE / IQR(y_true))
    * Skill_vs_persistence = 100 * (1 - MAE(model)/MAE(naive(tomorrow=today)))
- Prints raw and calibrated coverages for the quantile bands
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------- Paths --------------------
HERE = Path(__file__).resolve().parent
CSV_CANDIDATES = ["curiosity-weather.csv", "mars_weather_data_parsed.csv"]
MODELS_DIR = HERE / "models"
BUNDLE_PATHS = {
    "max_temp": MODELS_DIR / "model_max_temp.joblib",
    "min_temp": MODELS_DIR / "model_min_temp.joblib",
}


# -------------------- Metrics --------------------
def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / np.maximum(denom, eps))


def iqr(a):
    a = np.asarray(a, float)
    q75, q25 = np.percentile(a, [75, 25])
    return float(max(q75 - q25, 1e-6))


def accuracy_iqr(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return 100.0 * (1.0 - mae / iqr(y_true))


def skill_vs_persistence(y_true, y_pred, y_persist):
    mae_m = mean_absolute_error(y_true, y_pred)
    mae_n = mean_absolute_error(y_true, y_persist)
    return 100.0 * (1.0 - mae_m / max(mae_n, 1e-9)), mae_m, mae_n


# -------------------- Data helpers --------------------
def resolve_csv() -> Path:
    for name in CSV_CANDIDATES:
        p = HERE / name
        if p.exists():
            print(f"[info] Using CSV: {p.name}")
            return p
    csvs = sorted(HERE.glob("*.csv"))
    if csvs:
        print(f"[info] Using CSV found next to script: {csvs[0].name}")
        return csvs[0]
    raise FileNotFoundError("No CSV found next to the script.")


def coalesce_to(
    df_: pd.DataFrame, new_name: str, candidates: List[str], transform=None
) -> bool:
    for cand in candidates:
        if cand in df_.columns:
            df_[new_name] = df_[cand] if transform is None else transform(df_[cand])
            return True
    return False


def parse_hhmm(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .replace({"--": np.nan, "": np.nan, "None": np.nan, "none": np.nan})
    )
    return pd.to_datetime(s, format="%H:%M", errors="coerce").dt.hour


def to_tau(val):
    if pd.isna(val):
        return np.nan
    return {
        "Sunny": 0.35,
        "Light": 0.50,
        "Moderate": 0.70,
        "High": 0.90,
        "Very High": 1.10,
    }.get(str(val).strip().title(), 0.70)


# -------------------- Feature building (mirrors training) --------------------
def build_features(df: pd.DataFrame):
    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    coalesce_to(
        df,
        "min_temp",
        [
            "min_temp",
            "min_temp_c",
            "min_temperature",
            "min",
            "min_air_temp",
            "AT_min",
            "Tmin",
        ],
        to_num,
    )
    coalesce_to(
        df,
        "max_temp",
        [
            "max_temp",
            "max_temp_c",
            "max_temperature",
            "max",
            "max_air_temp",
            "AT_max",
            "Tmax",
        ],
        to_num,
    )
    coalesce_to(
        df,
        "pressure",
        ["pressure", "pressure_pa", "pressure_hpa", "PRE", "surface_pressure"],
        to_num,
    )
    coalesce_to(
        df,
        "wind_speed",
        [
            "wind_speed",
            "wind_speed_mps",
            "wind_mps",
            "wind",
            "wind_avg",
            "avg_wind",
            "HWS",
            "windspeed",
        ],
        to_num,
    )
    coalesce_to(df, "sunrise", ["sunrise", "sunrise_time", "sun_rise", "srise"])
    coalesce_to(df, "sunset", ["sunset", "sunset_time", "sun_set", "sset"])
    coalesce_to(df, "season", ["season", "martian_season", "szn"])
    coalesce_to(df, "wind_direction", ["wind_direction", "wind_dir", "wd", "HWD"])
    if not coalesce_to(df, "sol", ["sol", "Sol", "sol_id", "sol_number"], to_num):
        df = df.reset_index(drop=True)
        df["sol"] = df.index.astype(int)

    for c in ["sunrise", "sunset"]:
        if c not in df.columns:
            df[c] = np.nan
    df["sunrise_h"] = parse_hhmm(df["sunrise"]).astype("Int64")
    df["sunset_h"] = parse_hhmm(df["sunset"]).astype("Int64")
    df["day_len_h"] = (df["sunset_h"] - df["sunrise_h"]).astype("Float64").clip(lower=0)

    df["tau_proxy"] = df.get("atmo_opacity", pd.Series(np.nan, index=df.index)).map(
        to_tau
    )
    if "ls" in df.columns:
        df["ls"] = pd.to_numeric(df["ls"], errors="coerce")
        rad = np.deg2rad(df["ls"])
        df["sin_ls"], df["cos_ls"] = np.sin(rad), np.cos(rad)
    else:
        df["sin_ls"], df["cos_ls"] = np.nan, np.nan

    df["max_temp"] = df["max_temp"].clip(-120, 20)
    df["min_temp"] = df["min_temp"].clip(-140, 10)

    df = df.sort_values("sol").reset_index(drop=True)
    targets = [
        t
        for t in ["max_temp", "min_temp"]
        if t in df.columns and df[t].notna().sum() >= 2
    ]
    for t in targets:
        df[f"{t}__next"] = pd.to_numeric(df[t], errors="coerce").shift(-1)

    base_for_lags = ["min_temp", "max_temp", "pressure", "wind_speed", "tau_proxy"]
    for c in base_for_lags:
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_roll3"] = df[c].rolling(3, min_periods=1).mean()
        df[f"{c}_roll7"] = df[c].rolling(7, min_periods=1).mean()

    df["temp_grad"] = df["max_temp"] - df["min_temp"]
    df["tau_daylen"] = df["tau_proxy"] * df["day_len_h"]

    # Modeling frame
    target_cols = [f"{t}__next" for t in targets]
    df_model = df.dropna(subset=target_cols).copy()
    base_cats = [c for c in ["season", "wind_direction"] if c in df_model.columns]
    num_feats = (
        [
            "sol",
            "sunrise_h",
            "sunset_h",
            "day_len_h",
            "tau_proxy",
            "sin_ls",
            "cos_ls",
            "temp_grad",
            "tau_daylen",
        ]
        + [f"{c}_lag1" for c in base_for_lags]
        + [f"{c}_roll3" for c in base_for_lags]
        + [f"{c}_roll7" for c in base_for_lags]
    )
    feat_cols_all = base_cats + num_feats
    return df_model, targets, base_cats, num_feats, feat_cols_all


def mk_time_split(df_part: pd.DataFrame, feat_cols_all, base_cats, num_feats):
    X = df_part[feat_cols_all]
    y_map = {
        t: df_part[f"{t}__next"]
        for t in ["max_temp", "min_temp"]
        if f"{t}__next" in df_part.columns
    }
    n = len(df_part)
    cut = n if n < 3 else max(1, min(int(n * 0.85), n - 1))
    Xtr, Xte = X.iloc[:cut], X.iloc[cut:]
    ytr = {t: y_map[t].iloc[:cut] for t in y_map}
    yte = {t: y_map[t].iloc[cut:] for t in y_map}

    num_ok = [c for c in num_feats if c in Xtr.columns and Xtr[c].notna().any()]
    cats_ok = [c for c in base_cats if c in Xtr.columns and Xtr[c].notna().any()]
    feat_ok = cats_ok + num_ok
    Xtr = Xtr[feat_ok]
    Xte = Xte[feat_ok]
    return Xtr, Xte, ytr, yte, feat_ok


# -------------------- Evaluation --------------------
def evaluate_target(tname: str, bundle, df_model, Xte, yte):
    if tname not in yte:
        print(f"[{tname}] skipped (no target in data)")
        return

    # Align columns to the bundle's training columns
    cols = bundle["feat_cols"]
    Xte_aligned = Xte.reindex(columns=cols, fill_value=np.nan)

    # Predict quantiles
    p = {}
    for q in ("p10", "p50", "p90"):
        p[q] = bundle["models"][q].predict(Xte_aligned)

    # Apply saved bias model (if any)
    bias_model = bundle.get("bias_model", None)
    if bias_model is not None:
        bias = bias_model.predict(p["p50"])
        p["p50"] = p["p50"] + bias
        p["p10"] = p["p10"] + bias
        p["p90"] = p["p90"] + bias

    y_true = yte[tname].to_numpy()
    y_today = df_model.loc[Xte.index, tname].to_numpy()  # persistence baseline

    # Metrics
    mae = mean_absolute_error(y_true, p["p50"])
    r2p = r2_score(y_true, p["p50"]) * 100.0
    sm = smape(y_true, p["p50"])
    acc = accuracy_iqr(y_true, p["p50"])
    skill, mae_m, mae_n = skill_vs_persistence(y_true, p["p50"], y_today)

    cov_raw = ((y_true >= p["p10"]) & (y_true <= p["p90"])).mean() * 100.0

    # Calibrated coverage using saved scale
    s = float(bundle.get("quantile_scale", 1.0))
    hw = (p["p90"] - p["p10"]) / 2.0
    lo_cal, hi_cal = p["p50"] - s * hw, p["p50"] + s * hw
    cov_cal = ((y_true >= lo_cal) & (y_true <= hi_cal)).mean() * 100.0

    print(f"\n[eval] {tname} on {len(y_true)} test rows")
    print(f"  R²:                   {r2p:6.2f}%")
    print(f"  MAE:                  {mae:6.2f} °C")
    print(f"  sMAPE (ref):          {sm:6.2f}%")
    print(f"  Accuracy_IQR:         {acc:6.2f}%")
    print(
        f"  Skill_vs_persist:     {skill:6.2f}% (MAE_model={mae_m:.2f}, MAE_naive={mae_n:.2f})"
    )
    print(f"  Coverage raw:         {cov_raw:6.2f}%")
    print(f"  Coverage calibrated:  {cov_cal:6.2f}% (scale={s:.2f})")


def main():
    # 1) Load CSV and rebuild features
    csv_path = resolve_csv()
    df = pd.read_csv(
        csv_path,
        sep=None,
        engine="python",
        na_values=["--", "", "None", "none", "NA", "NaN", "null"],
    )
    df_model, targets, base_cats, num_feats, feat_cols_all = build_features(df)
    print(f"[info] rows total={len(df)} model_rows={len(df_model)} | targets={targets}")

    if not targets:
        raise RuntimeError("No usable targets (need ≥2 rows with values).")

    # 2) Make the same time-safe split as training
    Xtr, Xte, ytr, yte, feat_ok = mk_time_split(
        df_model, feat_cols_all, base_cats, num_feats
    )
    if len(Xte) == 0:
        raise RuntimeError("No test rows after split; need more data to evaluate.")

    # 3) Load bundles and evaluate each available target
    for tname, bpath in BUNDLE_PATHS.items():
        if not bpath.exists():
            print(f"[warn] Missing bundle for {tname}: {bpath.name} (skipping)")
            continue
        bundle = joblib.load(bpath)
        evaluate_target(tname, bundle, df_model, Xte, yte)


if __name__ == "__main__":
    main()
