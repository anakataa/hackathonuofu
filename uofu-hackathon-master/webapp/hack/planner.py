# webapp/hack/flaskapp/services/planner.py
# CHANGE IT ONCE THE DATA WILL BE AVAILABLE
"""
Risk-aware Mars day planner.

Inputs (typical):
- sol: int                  -> target sol number (for display/bookkeeping)
- tasks: List[dict]         -> task templates (see TASK SCHEMA below)
- forecast: dict            -> {"forecast": [hourly entries ...]} (optional)
- policy: dict              -> {"mode": "conservative"|"normal"|"aggressive"}

TASK SCHEMA (example):
{
  "id": "eva_sample_alpha",
  "duration_min": 90,
  "earliest_lmst": "08:00",     # optional
  "latest_lmst":   "18:00",     # optional
  "min_temp_c":    -35,         # only relevant if req_outside
  "max_wind_mps":  10,          # only relevant if req_outside
  "max_tau":       0.8,         # only relevant if req_outside
  "power_wh":      400,         # approximate energy budget for task
  "req_outside":   True,        # EVA/rover work
  "priority":      0.9,         # higher = earlier in schedule
  "cooldown_min":  30,          # gap after task
  "resource":      "EVA",       # "EVA" | "ROVER" | "INDOOR"
}

Output:
{
  "sol": <int>,
  "items": [
    {
      "task_id": "...",
      "start_lmst": "HH:MM",
      "end_lmst": "HH:MM",
      "risk_score": float,
      "because": [str, ...],
      "power_end_wh": float
    },
    ...
  ],
  "unscheduled": [task_id, ...],
  "score": float
}
"""

from __future__ import annotations
from typing import Dict, List, Tuple

# ---- Policy knobs (tweak quickly during the hackathon)
DEFAULT_RESERVE_SOC = 0.30  # keep 30% battery reserve
START_SOC_WH = 2000.0  # fake initial energy (Wh)
PV_CHARGE_WH_PER_HOUR = 250.0  # simple daytime charging model (Wh/h)
INDOOR_LOAD_WH_PER_HOUR = 120.0  # base habitat load (not modeled per-task)

RESOURCE_LIMITS = {
    "EVA": 2,  # number of EVA suits/crews usable concurrently
    "ROVER": 1,  # one rover
    "INDOOR": 99,  # many indoor activities in parallel (soft-limited by SoC only)
}


def _hhmm_to_hour(hhmm: str) -> int:
    if not hhmm:
        return None  # noqa
    h, m = hhmm.split(":")
    return int(h) + (1 if int(m) >= 30 else 0)


def _fmt_hour(h: int) -> str:
    return f"{h:02d}:00"


def _p(field: str, entry: dict, fallback=None):
    """
    Prefer quantiles when provided:
    - safety-critical: use p90(wind), p90(tau), p10(temp)
    Otherwise fall back to point estimate.
    """
    if field == "wind_mps":
        return entry.get("p90", {}).get("wind_mps", entry.get("wind_mps", fallback))
    if field == "tau":
        return entry.get("p90", {}).get("tau", entry.get("tau", fallback))
    if field == "temp_c":
        return entry.get("p10", {}).get("temp_c", entry.get("temp_c", fallback))
    return entry.get(field, fallback)


def _window_ok_for_outside(
    entry: dict, task: dict, policy_mode: str
) -> Tuple[bool, List[str], float]:
    """Check weather gates for EVA/ROVER tasks. Return (ok, because[], risk_score)."""
    because = []
    # thresholds (allow policy to tune conservatism)
    max_wind = task.get("max_wind_mps", 10)
    max_tau = task.get("max_tau", 0.8)
    min_temp = task.get("min_temp_c", -35)

    # more conservative = tighten gates a bit
    tighten = {"conservative": 0.9, "normal": 1.0, "aggressive": 1.1}.get(
        policy_mode, 1.0
    )

    wind = _p("wind_mps", entry, 0)
    tau = _p("tau", entry, 1.0)
    temp = _p("temp_c", entry, -100)

    wind_ok = wind <= max_wind * tighten
    tau_ok = tau <= max_tau * tighten
    temp_ok = temp >= min_temp * (2 - tighten)  # inverse-ish scaling

    if wind_ok:
        because.append(f"wind_p90={wind:.1f}≤{max_wind}")
    if tau_ok:
        because.append(f"tau≤{max_tau}")
    if temp_ok:
        because.append(f"temp_p10={temp:.1f}≥{min_temp}")

    ok = wind_ok and tau_ok and temp_ok

    # crude risk: normalize to [0,1], penalize proximity to limits
    wind_r = min(1.0, wind / max(max_wind, 0.1))
    tau_r = min(1.0, tau / max(max_tau, 0.01))
    temp_r = 1.0 - min(1.0, (temp - min_temp) / max(abs(min_temp) + 1e-6, 1.0))
    risk = 0.4 * wind_r + 0.35 * tau_r + 0.25 * temp_r

    return ok, because, float(risk)


def _resources_ok(
    resource_timeline: Dict[str, Dict[int, int]], res: str, hour: int, dur_h: int
) -> bool:
    cap = RESOURCE_LIMITS.get(res, 1)
    for h in range(hour, hour + dur_h):
        if resource_timeline.setdefault(res, {}).get(h, 0) >= cap:
            return False
    return True


def _book_resource(
    resource_timeline: Dict[str, Dict[int, int]], res: str, hour: int, dur_h: int
) -> None:
    for h in range(hour, hour + dur_h):
        resource_timeline.setdefault(res, {})
        resource_timeline[res][h] = resource_timeline[res].get(h, 0) + 1


def _update_soc_trace(
    soc_trace: Dict[int, float], start_h: int, end_h: int, task_wh: float
) -> Tuple[bool, float]:
    """Very simple SoC model: charge during 10–16h; consume task_wh spread uniformly."""
    # copy last value forward if missing
    if 0 not in soc_trace:
        soc_trace[0] = START_SOC_WH
    for h in range(1, 24):
        soc_trace.setdefault(h, soc_trace[h - 1])

    # availability windows for PV (fake)
    CHARGE_HOURS = set(range(10, 17))

    # compute per-hour load
    dur = max(1, end_h - start_h)
    task_per_hour = task_wh / dur

    # simulate and check reserve
    for h in range(start_h, end_h):
        soc = soc_trace[h]
        charge = PV_CHARGE_WH_PER_HOUR if h in CHARGE_HOURS else 0.0
        next_soc = soc + charge - (task_per_hour + INDOOR_LOAD_WH_PER_HOUR)
        if next_soc < START_SOC_WH * DEFAULT_RESERVE_SOC:
            return False, soc_trace[end_h - 1]
        soc_trace[h + 1] = next_soc
    return True, soc_trace[end_h]


def _hour_bounds(task: dict) -> Tuple[int, int]:
    earliest = _hhmm_to_hour(task.get("earliest_lmst") or "06:00") or 6
    latest = _hhmm_to_hour(task.get("latest_lmst") or "20:00") or 20
    return max(0, earliest), min(23, latest)


def generate_plan(payload: dict) -> dict:
    sol = int(payload.get("sol", 520))
    tasks = list(payload.get("tasks", []))
    forecast = payload.get(
        "forecast"
    )  # if omitted, caller can fetch from weather svc first
    policy = (payload.get("policy") or {}).get("mode", "normal")

    hours = (
        forecast["forecast"]
        if forecast and "forecast" in forecast
        else [
            # fallback: safe-ish dummy day
            {
                "hour": h,
                "temp_c": -30 + (h - 12) * 0.5,
                "wind_mps": 6,
                "tau": 0.6,
                "p90": {"wind_mps": 7, "tau": 0.65},
                "p10": {"temp_c": -38},
            }
            for h in range(24)
        ]
    )

    # sort tasks by priority (desc), then duration (desc)
    tasks.sort(
        key=lambda t: (float(t.get("priority", 0.5)), int(t.get("duration_min", 60))),
        reverse=True,
    )

    resource_timeline: Dict[str, Dict[int, int]] = {}
    soc_trace: Dict[int, float] = {}  # hour -> SoC(Wh)

    items, unscheduled = [], []

    for t in tasks:
        dur_h = max(1, int(round(t.get("duration_min", 60) / 60)))
        res = t.get("resource", "INDOOR")
        outside = bool(t.get("req_outside", res in ("EVA", "ROVER")))
        start_bound, end_bound = _hour_bounds(t)

        placed = False
        best_slot = None
        best_risk = 9e9
        because_best = []

        # try every feasible start hour
        for h in range(start_bound, max(start_bound, end_bound - dur_h + 1)):
            # weather gate (check each hour in the block)
            risk_sum = 0.0
            because_local = []
            ok_weather = True
            for hh in range(h, h + dur_h):
                entry = next((e for e in hours if e["hour"] == hh), None)
                if entry is None:
                    ok_weather = False
                    break
                if outside:
                    ok_h, because, risk = _window_ok_for_outside(entry, t, policy)
                    if not ok_h:
                        ok_weather = False
                        break
                    because_local.extend(because)
                    risk_sum += risk
            if not ok_weather:
                continue

            # resources
            if not _resources_ok(resource_timeline, res, h, dur_h):
                continue

            # power
            soc_copy = dict(soc_trace)
            ok_soc, soc_end = _update_soc_trace(
                soc_copy, h, h + dur_h, float(t.get("power_wh", 200))
            )
            if not ok_soc:
                continue

            # choose lowest risk slot
            if risk_sum < best_risk:
                best_risk = risk_sum
                best_slot = (h, h + dur_h, soc_copy, soc_end, because_local)

        if best_slot:
            s, e, soc_copy, soc_end, because_local = best_slot
            _book_resource(resource_timeline, res, s, dur_h)
            soc_trace.update(soc_copy)
            items.append(
                {
                    "task_id": t.get("id", "task"),
                    "start_lmst": _fmt_hour(s),
                    "end_lmst": _fmt_hour(e),
                    "risk_score": round(best_risk / dur_h, 3),
                    "because": sorted(set(because_local))[:4],  # keep it readable
                    "power_end_wh": round(soc_end, 1),
                }
            )
            # cooldown buffer
            cooldown_h = max(0, int(round(t.get("cooldown_min", 0) / 60)))
            for pad in range(e, min(24, e + cooldown_h)):
                _book_resource(resource_timeline, res, pad, 1)
            placed = True

        if not placed:
            unscheduled.append(t.get("id", "task"))

    # simple score: more high-priority minutes with lower risk is better
    score = 0.0
    for it in items:
        pr = next(
            (tt.get("priority", 0.5) for tt in tasks if tt.get("id") == it["task_id"]),
            0.5,
        )
        dur = int(it["end_lmst"][:2]) - int(it["start_lmst"][:2])
        score += pr * dur * (1.0 - it["risk_score"])

    return {
        "sol": sol,
        "items": items,
        "unscheduled": unscheduled,
        "score": round(score, 3),
    }
