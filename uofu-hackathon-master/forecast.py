# file: mars_forecast_3day.py
import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# ========== helpers ==========
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _parse_hhmm(series: pd.Series) -> pd.Series:
    s = (
        series.astype(str)
        .str.strip()
        .replace({"--": np.nan, "": np.nan, "None": np.nan, "none": np.nan})
    )
    return pd.to_datetime(s, format="%H:%M", errors="coerce").dt.hour


def _to_tau(val):
    if pd.isna(val):
        return np.nan
    return {
        "Sunny": 0.35,
        "Light": 0.50,
        "Moderate": 0.70,
        "High": 0.90,
        "Very High": 1.10,
    }.get(str(val).strip().title(), 0.70)


def _coalesce(df, new, cands, transform=None):
    for c in cands:
        if c in df.columns:
            df[new] = transform(df[c]) if transform else df[c]
            return True
    return False


def _coalesce_date(
    df,
    new="__date__",
    cands=(
        "date",
        "terrestrial_date",
        "earth_date",
        "timestamp",
        "datetime",
        "time",
        "Date",
    ),
    tz="America/Denver",
):
    """
    Build a tz-naive (but Denver-anchored) midnight date column in df[new], if any candidate exists.
    """
    for c in cands:
        if c in df.columns:
            d = pd.to_datetime(df[c], errors="coerce", utc=True)
            if d.notna().any():
                # tz-aware (UTC) -> convert to Denver -> normalize -> drop tz (naive)
                df[new] = d.dt.tz_convert(tz).dt.normalize().dt.tz_localize(None)
                return True
    return False


def build_features_from_raw(df_raw: pd.DataFrame, tz="America/Denver") -> pd.DataFrame:
    df = df_raw.copy()
    _coalesce_date(df, tz=tz)

    _coalesce(
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
        _to_num,
    )
    _coalesce(
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
        _to_num,
    )
    _coalesce(
        df,
        "pressure",
        ["pressure", "pressure_pa", "pressure_hpa", "PRE", "surface_pressure"],
        _to_num,
    )
    _coalesce(
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
        _to_num,
    )
    _coalesce(df, "sunrise", ["sunrise", "sunrise_time", "sun_rise", "srise"])
    _coalesce(df, "sunset", ["sunset", "sunset_time", "sun_set", "sset"])
    _coalesce(df, "season", ["season", "martian_season", "szn"])
    _coalesce(df, "wind_direction", ["wind_direction", "wind_dir", "wd", "HWD"])

    if not _coalesce(df, "sol", ["sol", "Sol", "sol_id", "sol_number"], _to_num):
        df = df.reset_index(drop=True)
        df["sol"] = df.index.astype(int)

    # times/daylight
    for c in ["sunrise", "sunset"]:
        if c not in df.columns:
            df[c] = np.nan
    df["sunrise_h"] = _parse_hhmm(df["sunrise"]).astype("Int64")
    df["sunset_h"] = _parse_hhmm(df["sunset"]).astype("Int64")
    df["day_len_h"] = (df["sunset_h"] - df["sunrise_h"]).astype("Float64").clip(lower=0)

    # tau & seasonality
    df["tau_proxy"] = (
        df["atmo_opacity"].map(_to_tau) if "atmo_opacity" in df.columns else np.nan
    )
    if "ls" in df.columns:
        df["ls"] = pd.to_numeric(df["ls"], errors="coerce")
        rad = np.deg2rad(df["ls"])
        df["sin_ls"], df["cos_ls"] = np.sin(rad), np.cos(rad)
    else:
        df["sin_ls"], df["cos_ls"] = np.nan, np.nan

    # stability clips
    df["max_temp"] = df["max_temp"].clip(-120, 20)
    df["min_temp"] = df["min_temp"].clip(-140, 10)

    # temporal order by sol; date is used separately just to pick "today"
    df = df.sort_values("sol").reset_index(drop=True)

    # lags/rolls
    base = ["min_temp", "max_temp", "pressure", "wind_speed", "tau_proxy"]
    for c in base:
        if c not in df.columns:
            df[c] = np.nan
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_roll3"] = df[c].rolling(3, min_periods=1).mean()
        df[f"{c}_roll7"] = df[c].rolling(7, min_periods=1).mean()

    df["temp_grad"] = df["max_temp"] - df["min_temp"]
    df["tau_daylen"] = df["tau_proxy"] * df["day_len_h"]
    return df


def _apply_iso_and_order(p10, p50, p90, iso):
    if iso is not None:
        p50_iso = float(iso.predict([p50])[0])
        d = p50_iso - p50
        p10, p50, p90 = p10 + d, p50_iso, p90 + d
    lo, hi = (p10, p90) if p10 <= p90 else (p90, p10)
    p50 = min(max(p50, lo), hi)
    return lo, p50, hi


def _predict_one(bundle, row_df: pd.DataFrame):
    models = bundle["models"]
    cols = bundle["feat_cols"]
    iso = bundle.get("bias_model", None)
    s = float(bundle.get("quantile_scale", 1.0))

    X = row_df[cols]
    a = float(models["p10"].predict(X)[0])
    b = float(models["p50"].predict(X)[0])
    c = float(models["p90"].predict(X)[0])

    p10, p50, p90 = _apply_iso_and_order(a, b, c, iso)
    hw = max((p90 - p10) / 2.0, 1e-6)
    return {
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "p10_cal": p50 - s * hw,
        "p90_cal": p50 + s * hw,
    }


# ========== forecast (tz-safe + sol advances + future dates) ==========
def forecast_next_n(
    df_raw: pd.DataFrame,
    n_days: int = 3,
    model_dir: str = "./models",
    start_date: str | None = None,
    tz: str = "America/Denver",
) -> pd.DataFrame:
    """
    Picks 'current world day' as the most recent row with __date__ <= today (Denver-local).
    If no date column exists, falls back to last-by-sol and stamps today's date (Denver).
    Returns next n days with calibrated ranges; printed dates are always 'today + 1..n'.
    """
    # 'today' as tz-naive but Denver-anchored to avoid tz comparison errors
    today = (
        pd.Timestamp.now(tz).normalize().tz_localize(None)
        if start_date is None
        else pd.Timestamp(start_date).normalize()
    )

    df_feat = build_features_from_raw(df_raw, tz=tz)
    if len(df_feat) == 0:
        raise ValueError("No rows in input data.")

    # Load bundles
    b_max = joblib.load(Path(model_dir) / "model_max_temp.joblib")
    b_min = joblib.load(Path(model_dir) / "model_min_temp.joblib")

    # Choose starting "today" (for features/rolls), but output dates will be anchored to 'today'
    has_date = "__date__" in df_feat.columns and df_feat["__date__"].notna().any()
    if has_date:
        df_dated = (
            df_feat[df_feat["__date__"].notna()]
            .sort_values(["__date__", "sol"])
            .reset_index(drop=True)
        )
        mask = df_dated["__date__"] <= today
        start_idx = int(df_dated.index[mask].max()) if mask.any() else 0
        last = df_dated.iloc[[start_idx]].copy()
        hist = df_dated.iloc[: start_idx + 1].copy()
    else:
        hist = df_feat.copy()
        last = hist.iloc[[-1]].copy()
        if "__date__" not in hist.columns:
            hist["__date__"] = pd.NaT
        hist.loc[last.index, "__date__"] = today

    has_ls = "ls" in df_raw.columns

    out_rows = []
    for step in range(1, n_days + 1):
        row = last.copy()

        # --- dates & sol ---
        # Output date anchored to 'today' (ensures FUTURE dates)
        next_date = today + pd.Timedelta(days=step)
        row.loc[:, "__date__"] = next_date

        # Advance sol (features expect next-sol prediction)
        if "sol" in row.columns and pd.notna(row["sol"].iloc[0]):
            row.loc[:, "sol"] = int(row["sol"].iloc[0]) + 1

        # Optional: advance ls
        if has_ls and "ls" in row.columns and pd.notna(row["ls"].iloc[0]):
            new_ls = (float(row["ls"].iloc[0]) + 360.0 / 668.6) % 360.0
            row.loc[:, "ls"] = new_ls
            rad = np.deg2rad(new_ls)
            row.loc[:, "sin_ls"], row.loc[:, "cos_ls"] = np.sin(rad), np.cos(rad)

        # Predict using updated 'row' features
        pred_max = _predict_one(b_max, row)
        pred_min = _predict_one(b_min, row)

        out_rows.append(
            {
                "date": next_date.normalize(),
                "sol": (
                    int(row["sol"].iloc[0])
                    if "sol" in row.columns and pd.notna(row["sol"].iloc[0])
                    else np.nan
                ),
                "max_p10_cal": pred_max["p10_cal"],
                "max_p90_cal": pred_max["p90_cal"],
                "min_p10_cal": pred_min["p10_cal"],
                "min_p90_cal": pred_min["p90_cal"],
                "max_p50": pred_max["p50"],
                "min_p50": pred_min["p50"],  # kept if needed later
            }
        )

        # carry state forward (copy from row so sol/date/ls advance)
        next_today = row.copy()
        next_today.loc[:, "__date__"] = next_date
        next_today.loc[:, "max_temp"] = pred_max["p50"]
        next_today.loc[:, "min_temp"] = pred_min["p50"]
        next_today.loc[:, "temp_grad"] = next_today["max_temp"] - next_today["min_temp"]
        next_today.loc[:, "tau_daylen"] = (
            next_today["tau_proxy"] * next_today["day_len_h"]
        )

        hist = pd.concat([hist, next_today], ignore_index=True)
        for c in ["min_temp", "max_temp", "pressure", "wind_speed", "tau_proxy"]:
            hist[f"{c}_lag1"] = hist[c].shift(1)
            hist[f"{c}_roll3"] = hist[c].rolling(3, min_periods=1).mean()
            hist[f"{c}_roll7"] = hist[c].rolling(7, min_periods=1).mean()

        last = hist.iloc[[-1]].copy()

    return pd.DataFrame(out_rows)


# ========== pretty printing ==========
MINUS = "−"  # U+2212


def _fmt_range(a, b) -> str:
    # force numeric & ordered so it always prints [lower, upper] °C
    lo = float(a)
    hi = float(b)
    if lo > hi:
        lo, hi = hi, lo
    lo_s = f"{lo:.2f}".replace("-", MINUS)
    hi_s = f"{hi:.2f}".replace("-", MINUS)
    return f"[{lo_s}, {hi_s}] °C"


if __name__ == "__main__":
    # Load your latest CSV (same schema as training)
    df_raw = pd.read_csv(
        "mars_weather_data_parsed.csv",
        sep=None,
        engine="python",
        na_values=["--", "", "None", "none", "NA", "NaN", "null"],
    )

    # Run forecast (Denver-local "today"; set start_date="2025-10-24" to override)
    fc = forecast_next_n(
        df_raw, n_days=3, model_dir="./models", start_date=None, tz="America/Denver"
    )

    # Print calibrated ranges only, always ordered: [lower, upper] °C
    for _, r in fc.iterrows():
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else "N/A"
        sol_str = f" (sol {int(r['sol'])})" if pd.notna(r["sol"]) else ""
        print(
            f"{date_str}{sol_str}  "
            f"Max {_fmt_range(r['max_p10_cal'], r['max_p90_cal'])}  |  "
            f"Min {_fmt_range(r['min_p10_cal'], r['min_p90_cal'])}"
        )
