#!/usr/bin/env python3
"""
Global models + calibrated quantile bands (no per-season models).

Safe baseline defaults:
- Isotonic OFF by default (enable to test); direct-fit with guardrail if ON
- EWM / Delta features OFF by default (toggle to A/B)
- Quantile order enforced; half-width clamped
- Bisection calibration to target coverage (~80%)
- Comparable metrics: Accuracy_IQR, Skill_vs_persistence; sMAPE (reference)

Outputs:
- ./models/model_max_temp.joblib
- ./models/model_min_temp.joblib
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

# -------------------- Config --------------------
HERE = Path(__file__).resolve().parent
CSV_CANDIDATES = ["curiosity-weather.csv", "mars_weather_data_parsed.csv"]
MODELS_DIR = HERE / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COVERAGE = 0.80

# Baseline-safe toggles (start OFF; turn on one-by-one later if they help)
APPLY_ISO_MAX = False  # safer direct-fit iso with guardrail (enable to test)
APPLY_ISO_MIN = False
USE_EWM = False
USE_DELTAS = False

# GBM params (your original max; slightly larger for min is optional later)
GBM_MAX = dict(
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=3,
    min_samples_leaf=40,
    random_state=42,
)
GBM_MIN = dict(
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=3,
    min_samples_leaf=40,
    random_state=42,
)


# -------------------- Metrics --------------------
def smape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return 100.0 * np.mean(np.abs(y_pred - y_true) / np.maximum(denom, eps))


def iqr(a):
    a = np.asarray(a, dtype=float)
    q75, q25 = np.percentile(a, [75, 25])
    return float(max(q75 - q25, 1e-6))


def accuracy_iqr(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    return 100.0 * (1.0 - mae / iqr(y_true))


def skill_vs_persistence(y_true, y_pred, y_persist):
    mae_m = mean_absolute_error(y_true, y_pred)
    mae_n = mean_absolute_error(y_true, y_persist)
    return 100.0 * (1.0 - mae_m / max(mae_n, 1e-9)), mae_m, mae_n


# -------------------- Helpers --------------------
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


def make_gbm(alpha: float, pre: ColumnTransformer, for_min: bool):
    params = GBM_MIN if for_min else GBM_MAX
    gb = GradientBoostingRegressor(loss="quantile", alpha=alpha, **params)
    return Pipeline([("pre", pre), ("gb", gb)])


def coverage_for_scale(y_true, p50, halfwidth, s):
    lo, hi = p50 - s * halfwidth, p50 + s * halfwidth
    return ((y_true >= lo) & (y_true <= hi)).mean()


def calibrate_quantiles_bisect(
    y_true, p10, p50, p90, target=0.80, lo=0.3, hi=2.5, iters=28
):
    """Bisection on interval scale to achieve target coverage."""
    hw = np.maximum((p90 - p10) / 2.0, 1e-6)
    c_lo = coverage_for_scale(y_true, p50, hw, lo)
    c_hi = coverage_for_scale(y_true, p50, hw, hi)
    tries = 0
    while not (c_lo <= target <= c_hi) and tries < 12:
        if c_lo > target:
            lo *= 0.7
            c_lo = coverage_for_scale(y_true, p50, hw, lo)
        if c_hi < target:
            hi *= 1.3
            c_hi = coverage_for_scale(y_true, p50, hw, hi)
        tries += 1
    s_lo, s_hi = lo, hi
    for _ in range(iters):
        mid = 0.5 * (s_lo + s_hi)
        if coverage_for_scale(y_true, p50, hw, mid) < target:
            s_lo = mid
        else:
            s_hi = mid
    return float(0.5 * (s_lo + s_hi))


def enforce_quantile_order(p10, p50, p90):
    """Ensure p10 ≤ p50 ≤ p90 elementwise."""
    p_lo = np.minimum(p10, p90)
    p_hi = np.maximum(p10, p90)
    p_mid = np.clip(p50, p_lo, p_hi)
    return p_lo, p_mid, p_hi


def diag(y_true, p10_raw, p50_raw, p90_raw, p10, p50, p90, label):
    def cov(lo, mid, hi):
        return np.mean((y_true >= lo) & (y_true <= hi)) * 100.0

    hw_raw = (p90_raw - p10_raw) / 2.0
    hw_iso = (p90 - p10) / 2.0
    print(f"[{label}] diag: n={len(y_true)}")
    print(f"  corr(p50_raw,y) = {np.corrcoef(p50_raw, y_true)[0,1]:.3f}")
    print(f"  MAE_raw         = {np.mean(np.abs(p50_raw - y_true)):.2f}°C")
    print(f"  MAE_after_iso   = {np.mean(np.abs(p50 - y_true)):.2f}°C")
    print(
        f"  cross_raw       = {np.mean(p90_raw < p10_raw):.3f}  cross_after = {np.mean(p90 < p10):.3f}"
    )
    print(
        f"  med halfwidth   raw={np.median(hw_raw):.2f}  after={np.median(hw_iso):.2f}"
    )
    print(
        f"  cov_raw raw={cov(p10_raw,p50_raw,p90_raw):.2f}%  after={cov(p10,p50,p90):.2f}%"
    )


def pack_bundle(
    models: Dict[str, Pipeline],
    feat_cols,
    cats,
    nums,
    quantile_scale: float,
    bias_model: Optional[IsotonicRegression] = None,
    gb_params: Optional[dict] = None,
):
    return {
        "models": models,
        "feat_cols": feat_cols,
        "present_cats": cats,
        "num_feats": nums,
        "quantile_scale": quantile_scale,
        "coverage_target": TARGET_COVERAGE,
        "bias_model": bias_model,  # None if not used
        "gb_params": gb_params,
    }


# -------------------- Load & features --------------------
CSV_PATH = resolve_csv()
df = pd.read_csv(
    CSV_PATH,
    sep=None,
    engine="python",
    na_values=["--", "", "None", "none", "NA", "NaN", "null"],
)
print("[debug] raw columns:", list(df.columns))

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

# Times & daylight
for c in ["sunrise", "sunset"]:
    if c not in df.columns:
        df[c] = np.nan
df["sunrise_h"] = parse_hhmm(df["sunrise"]).astype("Int64")
df["sunset_h"] = parse_hhmm(df["sunset"]).astype("Int64")
df["day_len_h"] = (df["sunset_h"] - df["sunrise_h"]).astype("Float64").clip(lower=0)

# Dust proxy + seasonality
df["tau_proxy"] = df.get("atmo_opacity", pd.Series(np.nan, index=df.index)).map(to_tau)
if "ls" in df.columns:
    df["ls"] = pd.to_numeric(df["ls"], errors="coerce")
    rad = np.deg2rad(df["ls"])
    df["sin_ls"], df["cos_ls"] = np.sin(rad), np.cos(rad)
else:
    df["sin_ls"], df["cos_ls"] = np.nan, np.nan

# Stability clips
df["max_temp"] = df["max_temp"].clip(-120, 20)
df["min_temp"] = df["min_temp"].clip(-140, 10)

# Targets
df = df.sort_values("sol").reset_index(drop=True)
targets = [
    t for t in ["max_temp", "min_temp"] if t in df.columns and df[t].notna().sum() >= 2
]
for t in targets:
    df[f"{t}__next"] = pd.to_numeric(df[t], errors="coerce").shift(-1)
if not targets:
    raise RuntimeError("No usable targets (need ≥2 rows with values).")

# Lags & rolling (proven)
base_for_lags = ["min_temp", "max_temp", "pressure", "wind_speed", "tau_proxy"]
for c in base_for_lags:
    if c not in df.columns:
        df[c] = np.nan
    df[f"{c}_lag1"] = df[c].shift(1)
    df[f"{c}_roll3"] = df[c].rolling(3, min_periods=1).mean()
    df[f"{c}_roll7"] = df[c].rolling(7, min_periods=1).mean()

# Optional EWMs
extra_feats = []
if USE_EWM:
    for c in base_for_lags:
        df[f"{c}_ewm3"] = df[c].ewm(span=3, adjust=False, min_periods=1).mean()
    extra_feats += [f"{c}_ewm3" for c in base_for_lags]

# Interactions + optional deltas
df["temp_grad"] = df["max_temp"] - df["min_temp"]
df["tau_daylen"] = df["tau_proxy"] * df["day_len_h"]
if USE_DELTAS:
    df["d_min1"] = df["min_temp"] - df["min_temp_lag1"]
    df["d_pre1"] = df["pressure"] - df["pressure_lag1"]
    extra_feats += ["d_min1", "d_pre1"]

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
    + extra_feats
)
feat_cols_all = base_cats + num_feats

print(f"[info] rows total={len(df)} model_rows={len(df_model)}")


# ---------- Split, prune, preprocessor ----------
def mk_pre_and_frames(df_part: pd.DataFrame):
    X = df_part[feat_cols_all]
    y_map = {t: df_part[f"{t}__next"] for t in targets}
    n = len(df_part)
    cut = n if n < 3 else max(1, min(int(n * 0.85), n - 1))
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train = {t: y_map[t].iloc[:cut] for t in targets}
    y_test = {t: y_map[t].iloc[cut:] for t in targets}

    num_ok = [c for c in num_feats if c in X_train.columns and X_train[c].notna().any()]
    cats_ok = [
        c for c in base_cats if c in X_train.columns and X_train[c].notna().any()
    ]
    feat_ok = cats_ok + num_ok
    X_train = X_train[feat_ok]
    X_test = X_test[feat_ok]

    pre = ColumnTransformer(
        [
            (
                "num",
                SimpleImputer(strategy="median"),
                [c for c in num_ok if c in X_train.columns],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cats_ok,
            ),
        ],
        remainder="drop",
    )
    return pre, feat_ok, cats_ok, num_ok, X_train, X_test, y_train, y_test


pre, feat_cols, cats, nums, Xtr, Xte, ytr, yte = mk_pre_and_frames(df_model)


# ---------- Train & save per-target ----------
def train_and_save(tname: str, for_min: bool, apply_iso: bool, gb_params_name: str):
    if tname not in targets:
        return
    models: Dict[str, Pipeline] = {}
    for alpha, tag in [(0.10, "p10"), (0.50, "p50"), (0.90, "p90")]:
        m = make_gbm(alpha, pre, for_min=for_min)
        m.fit(Xtr, ytr[tname])
        models[tag] = m

    s_scale = 1.0
    bias_model = None

    if len(Xte) > 0:
        cols = feat_cols
        Xte_aligned = Xte[cols]
        p10_raw = models["p10"].predict(Xte_aligned)
        p50_raw = models["p50"].predict(Xte_aligned)
        p90_raw = models["p90"].predict(Xte_aligned)
        y_true = yte[tname].to_numpy()
        y_today = df_model.loc[Xte.index, tname].to_numpy()  # persistence baseline

        # Safe isotonic (direct fit) with guardrail
        if apply_iso:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p50_raw, y_true)  # learn y ≈ f(p50_raw)
            p50_iso = iso.predict(p50_raw)
            delta = p50_iso - p50_raw
            p10_iso = p10_raw + delta
            p90_iso = p90_raw + delta
            # enforce order
            p10_iso, p50_iso, p90_iso = enforce_quantile_order(
                p10_iso, p50_iso, p90_iso
            )
            # keep only if it helps MAE
            mae_raw = mean_absolute_error(y_true, p50_raw)
            mae_iso = mean_absolute_error(y_true, p50_iso)
            if mae_iso <= mae_raw:
                p10, p50, p90 = p10_iso, p50_iso, p90_iso
                bias_model = iso
            else:
                p10, p50, p90 = p10_raw, p50_raw, p90_raw
                bias_model = None
        else:
            p10, p50, p90 = p10_raw, p50_raw, p90_raw

        # Final enforcement + diagnostics
        p10, p50, p90 = enforce_quantile_order(p10, p50, p90)
        diag(y_true, p10_raw, p50_raw, p90_raw, p10, p50, p90, label=tname)

        # Metrics
        mae = mean_absolute_error(y_true, p50)
        r2p = r2_score(y_true, p50) * 100.0
        acc = accuracy_iqr(y_true, p50)
        sm = smape(y_true, p50)
        skill, mae_m, mae_n = skill_vs_persistence(y_true, p50, y_today)

        # Coverage
        s_scale = calibrate_quantiles_bisect(y_true, p10, p50, p90, TARGET_COVERAGE)
        hw = np.maximum((p90 - p10) / 2.0, 1e-6)
        lo_cal, hi_cal = p50 - s_scale * hw, p50 + s_scale * hw
        cov_raw = ((y_true >= p10) & (y_true <= p90)).mean() * 100.0
        cov_cal = ((y_true >= lo_cal) & (y_true <= hi_cal)).mean() * 100.0

        print(
            f"\n[{tname}] R²: {r2p:7.2f}% | MAE: {mae:6.2f} °C | sMAPE (ref): {sm:6.2f}%"
        )
        print(
            f"        Accuracy_IQR: {acc:6.2f}% | Skill_vs_persist: {skill:6.2f}% "
            f"(MAE_model={mae_m:.2f}, MAE_naive={mae_n:.2f})"
        )
        print(
            f"        Coverage raw: {cov_raw:6.2f}% | Coverage calibrated: {cov_cal:6.2f}% (scale={s_scale:.2f}) "
            f"{'(iso kept)' if bias_model is not None else '(iso skipped)'}"
        )

    # Save bundle
    bundle = pack_bundle(
        models=models,
        feat_cols=feat_cols,
        cats=cats,
        nums=nums,
        quantile_scale=s_scale,
        bias_model=bias_model,
        gb_params=GBM_MIN if for_min else GBM_MAX,
    )
    out_path = MODELS_DIR / f"model_{tname}.joblib"
    joblib.dump(bundle, out_path)
    print(f"Saved {out_path}")


# Train/save both targets
train_and_save(
    "max_temp", for_min=False, apply_iso=APPLY_ISO_MAX, gb_params_name="GBM_MAX"
)
train_and_save(
    "min_temp", for_min=True, apply_iso=APPLY_ISO_MIN, gb_params_name="GBM_MIN"
)

# ---------- Demo prediction on last row ----------
if len(df_model) > 0:
    last_feat = df_model.iloc[[-1]][feat_cols].to_dict("records")[0]
    preview = {}
    for tname in ["max_temp", "min_temp"]:
        bpath = MODELS_DIR / f"model_{tname}.joblib"
        if not bpath.exists():
            continue
        bundle = joblib.load(bpath)
        cols = bundle["feat_cols"]
        row = pd.DataFrame([last_feat])[cols]
        preds = {
            q: float(bundle["models"][q].predict(row)[0]) for q in ("p10", "p50", "p90")
        }
        iso = bundle.get("bias_model", None)
        if iso is not None:
            p50_iso = float(iso.predict([preds["p50"]])[0])
            delta = p50_iso - preds["p50"]
            preds["p50"] = p50_iso
            preds["p10"] += delta
            preds["p90"] += delta
        preds["p10"], preds["p50"], preds["p90"] = enforce_quantile_order(
            preds["p10"], preds["p50"], preds["p90"]
        )
        s = bundle.get("quantile_scale", 1.0)
        hw = max((preds["p90"] - preds["p10"]) / 2.0, 1e-6)
        preds["p10_cal"] = preds["p50"] - s * hw
        preds["p90_cal"] = preds["p50"] + s * hw
        preview[tname] = preds
    print("\n[demo] prediction on last row (bias-corrected & calibrated):")
    print(preview)
else:
    print("[warn] No rows after target construction.")
