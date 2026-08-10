"""Trader strategy v2 -- Screener quality_ic_tilt ensemble (8 factors) + regime overlay.

Cross-sectional factor composite (CS rank -> z-score -> winsorize 3 sigma, dir +1),
fully-invested 15-asset long-only target, one atomic rebalance per 10-trading-day block.
Risk-off regime tilts toward defensive tradable assets (XAU/US10Y/CN10Y); never cash.
Signals come from persisted factor artifacts aligned to the visible trading date.
"""
import json
import math
from datetime import date as _date
from pathlib import Path

from alphacrafter.sim.utils import (
    get_account_dict,
    get_index_daily_data,
    get_stock_daily_data,
    rebalance_to_weights,
    register_hook,
)

BASE = Path(__file__).parent
ENSEMBLE_PATH = BASE / "factor_ensemble.json"
DATE_PATH = BASE.parent / "persistent" / "date.json"

ONLINE_START = "2026-07-16"
BLOCK = 10
CAP = 0.17
FLOOR = 0.012
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
AGGRESSIVE = {"SOX", "NDX", "ETH", "BTC", "000688.SH", "N225"}
EMBEDDED = {"mom_20d_skip5", "range_pos_252", "spx_corr60"}
ARTIFACT_START = "2020-01-01"


def _load_ensemble():
    try:
        payload = json.loads(ENSEMBLE_PATH.read_text())
    except (OSError, ValueError, TypeError):
        return []
    out = []
    for item in payload.get("selected_factors", []):
        if not isinstance(item, dict):
            continue
        fid = item.get("factor_id")
        w = float(item.get("weight", 0.0) or 0.0)
        if not fid or not math.isfinite(w) or w <= 0.0:
            continue
        out.append({"factor_id": str(fid), "weight": w,
                    "direction": int(item.get("direction", 1) or 1)})
    return out


def _signal_row(fid, row_idx, n_assets):
    """Return the signal vector for a factor at the given artifact row index."""
    if fid in EMBEDDED:
        try:
            art = json.loads((BASE / "factors" / f"{fid}.json").read_text())["signal_artifact"]
            dates = art.get("dates", [])
            values = art.get("values", [])
            if not dates or not values:
                return None
            idx = min(row_idx, len(values) - 1)
            return [float(x) if x is not None else float("nan") for x in values[idx]]
        except (OSError, ValueError, KeyError, TypeError):
            return None
    try:
        import numpy as np
        arr = np.load(BASE / "factors" / f"{fid}.signal.npy", allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] < n_assets:
            return None
        row = arr[min(row_idx, arr.shape[0] - 1)]
        return [float(x) if x is not None and not (isinstance(x, float) and math.isnan(x))
                else float("nan") for x in row[:n_assets]]
    except (OSError, ValueError, TypeError):
        return None


def _rank_z(vals):
    """Cross-sectional rank -> [0,1] -> z-score -> winsorize 3 sigma."""
    n = len(vals)
    valid = sorted((v, i) for i, v in enumerate(vals) if v == v)
    ranks = [0.5] * n
    nv = len(valid)
    for k, (_, i) in enumerate(valid):
        ranks[i] = k / max(1, nv - 1)
    mean = sum(ranks) / n
    var = sum((x - mean) ** 2 for x in ranks) / n
    sd = math.sqrt(var) if var > 1e-14 else 1e-12
    return [max(-3.0, min(3.0, (x - mean) / sd)) for x in ranks]


def _closes(assets, n=130):
    closes = {}
    for a in assets:
        try:
            df = get_stock_daily_data(a, days=n)
        except Exception:
            df = None
        if df is not None and "close" in df and len(df) >= 62:
            closes[a] = df["close"].astype(float)
    return closes


def _regime(closes, assets):
    """Return risk score R in [0,1] using only data visible at decision date."""
    rets = {}
    for a, c in closes.items():
        s = c.pct_change().dropna()
        if len(s) >= 60:
            rets[a] = s
    if len(rets) < 8:
        return 0.5, 20.0, 0.0, 0.0
    panel = __import__("pandas").concat(rets, axis=1, join="inner").dropna()
    market = panel.mean(axis=1).tail(60)
    m20 = float(market.tail(20).mean()) if len(market) >= 20 else 0.0
    disp20 = float(panel.tail(20).std(axis=1).mean()) if len(panel) >= 20 else 0.0
    vix_level = 20.0
    try:
        vf = get_index_daily_data("VIX", days=40)
        if vf is not None and "close" in vf and len(vf) >= 2:
            vix_level = float(vf["close"].iloc[-1])
    except Exception:
        pass
    vix_comp = max(0.0, min(1.0, (vix_level - 15.0) / 20.0))
    trend_comp = max(0.0, min(1.0, -m20 / 0.04))
    r = 0.6 * vix_comp + 0.4 * trend_comp
    return r, vix_level, m20, disp20


def _fit_weights(pref, cap=CAP, floor=FLOOR):
    """Iterative cap/floor normalization of a non-negative preference vector."""
    w = {a: max(0.0, float(x)) for a, x in pref.items()}
    for _ in range(300):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess > 1e-12:
            room = [a for a in w if w[a] < cap - 1e-12]
            if room:
                den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
                for a in room:
                    w[a] += excess * (max(0.0, pref.get(a, 0.0)) / den if den > 1e-12 else 1.0 / len(room))
        for a in w:
            w[a] = min(cap, w[a])
        short = sum(max(0.0, floor - x) for x in w.values())
        if short > 1e-12:
            donors = [a for a in w if w[a] > floor + 1e-12]
            avail = sum(w[a] - floor for a in donors)
            if avail > 1e-12:
                for a in donors:
                    w[a] -= short * (w[a] - floor) / avail
        for a in w:
            w[a] = max(0.0, w[a])
        if excess <= 1e-12 and short <= 1e-12:
            break
    total = sum(w.values())
    if total <= 0.0:
        n = len(w)
        return {a: 1.0 / n for a in w}
    return {a: x / total for a, x in w.items()}


def build_target(assets, date_state, ensemble):
    """Pure computation of (weights, forecast_returns, factor_ids, meta)."""
    trading_days = date_state.get("trading_days", [])
    visible = date_state.get("visible_through", date_state.get("current_date"))
    if ARTIFACT_START not in trading_days or visible not in trading_days:
        return None
    row_idx = trading_days.index(visible) - trading_days.index(ARTIFACT_START)
    if row_idx < 0:
        row_idx = 0

    n = len(assets)
    z = [0.0] * n
    used = []
    for fac in ensemble:
        row = _signal_row(fac["factor_id"], row_idx, n)
        if row is None:
            continue
        zz = _rank_z(row)
        z = [a + fac["weight"] * fac["direction"] * b for a, b in zip(z, zz)]
        used.append(fac["factor_id"])
    if not used:
        return None

    mean = sum(z) / n
    var = sum((x - mean) ** 2 for x in z) / n
    sd = math.sqrt(var) if var > 1e-14 else 1e-12
    z_std = [(x - mean) / sd for x in z]

    # regime overlay
    closes = _closes(assets)
    risk, vix, m20, disp = _regime(closes, assets)
    delta = 0.14 * risk

    # softmax base weights
    mx = max(z_std)
    exps = [math.exp(x - mx) for x in z_std]
    den = sum(exps)
    base = {a: exps[i] / den for i, a in enumerate(assets)}

    pref = {}
    for i, a in enumerate(assets):
        if a in DEFENSIVE:
            pref[a] = base[a] + delta / len(DEFENSIVE)
        else:
            pref[a] = base[a] * (1.0 - delta)
    weights = _fit_weights(pref)

    # deterministic forecast returns (10-day horizon): z * daily vol * sqrt(10)
    sigma = {}
    for a in assets:
        c = closes.get(a)
        if c is not None and len(c) >= 21:
            s = c.pct_change().dropna().tail(20)
            v = float(s.std()) if len(s) >= 5 else 0.01
            sigma[a] = v if v > 1e-6 else 0.01
        else:
            sigma[a] = 0.01
    forecast = {a: z_std[i] * sigma[a] * math.sqrt(10.0) for i, a in enumerate(assets)}
    forecast = {a: (max(-0.25, min(0.25, v)) if v == v else 0.0) for a, v in forecast.items()}

    meta = {"risk": risk, "vix": vix, "m20": m20, "disp": disp,
            "n_factors": len(used), "z": dict(zip(assets, z_std))}
    return weights, forecast, used, meta


@register_hook
def strategy_hook():
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    try:
        date_state = json.loads(DATE_PATH.read_text())
    except (OSError, ValueError, TypeError):
        return
    current = date_state.get("current_date", "")
    if current < ONLINE_START:
        return  # warm-up: capital frozen, no holdings
    trading_days = date_state.get("trading_days", [])
    weekdays = [x for x in trading_days if _date.fromisoformat(x).weekday() < 5]
    if current not in weekdays:
        return
    k = weekdays.index(current) - weekdays.index(ONLINE_START)
    if k % BLOCK != 0:
        return  # not the first day of a 10-trading-day block

    ensemble = _load_ensemble()
    if not ensemble:
        return
    built = build_target(assets, date_state, ensemble)
    if built is None:
        return
    weights, forecast, used, meta = built
    total = sum(weights.values())
    if not (math.isfinite(total) and abs(total - 1.0) < 1e-6):
        return
    if any(not math.isfinite(weights[a]) or weights[a] < 0.0 for a in assets):
        return
    rebalance_to_weights(
        weights,
        forecast_returns=forecast,
        factor_ids=used[:10],
        horizon_days=BLOCK,
    )
