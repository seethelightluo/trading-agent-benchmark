import json
from pathlib import Path
import numpy as np
from alphacrafter.sim.utils import (
    register_hook, get_account_dict, get_stock_daily_data, rebalance_to_weights
)

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
# Screener ensemble: medium-term trend is intentionally dominant in the bear regime.
FACTORS = {
    "mom40": .18, "mom20": .20, "down20": .12, "peer": .20,
    "clv": .15, "idio": .08, "range": .07,
}
MIN_W, MAX_W, CADENCE = .02, .14, 10
last_decision = None


def rank01(vals):
    good = {s: float(v) for s, v in vals.items() if np.isfinite(v)}
    out = {s: .5 for s in UNIVERSE}
    if len(good) < 2:
        return out
    a = np.asarray(list(good.values()))
    lo, hi = np.quantile(a, [.05, .95])
    clipped = {s: min(hi, max(lo, v)) for s, v in good.items()}
    ordered = sorted(clipped, key=clipped.get)
    for i, s in enumerate(ordered):
        out[s] = (i + 1.) / len(ordered)
    return out


def project_weights(raw):
    # Iterative capped-simplex projection; always returns all 15 names and sum 1.
    w = {s: max(MIN_W, min(MAX_W, float(raw.get(s, MIN_W)))) for s in UNIVERSE}
    for _ in range(100):
        gap = 1.0 - sum(w.values())
        if abs(gap) < 1e-10:
            break
        free = [s for s in UNIVERSE if MIN_W + 1e-12 < w[s] < MAX_W - 1e-12]
        if not free:
            free = [s for s in UNIVERSE if (gap > 0 and w[s] < MAX_W) or (gap < 0 and w[s] > MIN_W)]
        if not free:
            break
        step = gap / len(free)
        for s in free:
            w[s] = max(MIN_W, min(MAX_W, w[s] + step))
    # Feasible bounds make residual numerical error negligible; normalize defensively.
    total = sum(w.values())
    return {s: w[s] / total for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global last_decision
    account = get_account_dict()
    data = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=145)
        if df is None or len(df) < 45:
            continue
        df = df.sort_values("date").reset_index(drop=True)
        c = np.asarray(df["close"], float)
        h = np.asarray(df["high"], float)
        low = np.asarray(df["low"], float)
        r = c[1:] / np.maximum(c[:-1], 1e-12) - 1.
        data[s] = (c, h, low, r, str(df.iloc[-1]["date"]))
    if len(data) < 12:
        return
    decision_date = max(v[4] for v in data.values())
    if last_decision is not None:
        elapsed = (np.datetime64(decision_date, "D") - np.datetime64(last_decision, "D")) / np.timedelta64(1, "D")
        if elapsed < CADENCE:
            return

    f = {k: {} for k in FACTORS}
    invvol = {}
    recent = {}
    for s, (c, h, low, r, _) in data.items():
        vol = max(float(np.std(r[-20:])), .008)
        invvol[s] = 1. / vol
        recent[s] = c[-1] / max(c[-6], 1e-12) - 1.
        f["mom20"][s] = (c[-1] / max(c[-21], 1e-12) - 1.) / vol
        f["mom40"][s] = (c[-1] / max(c[-41], 1e-12) - 1.) / max(float(np.std(r[-40:])), .008)
        # Downside-adjusted trend: reward return while penalizing negative-return risk.
        neg = r[-20:][r[-20:] < 0]
        f["down20"][s] = (c[-1] / max(c[-21], 1e-12) - 1.) / max(float(np.mean(np.abs(neg))) if len(neg) else .008, .008)
        spread = np.maximum(h[-5:] - low[-5:], 1e-12)
        f["clv"][s] = float(np.mean((2*c[-5:] - h[-5:] - low[-5:]) / spread))
        f["idio"][s] = -float(np.mean(r[-2:]))
        f["range"][s] = -float((c[-1] - low[-6:].min()) / max(h[-6:].max() - low[-6:].min(), 1e-12))
    median_recent = float(np.median(list(recent.values())))
    f["peer"] = {s: v - median_recent for s, v in recent.items()}
    ranks = {k: rank01(v) for k, v in f.items()}
    score = {s: sum(FACTORS[k] * ranks[k][s] for k in FACTORS) for s in UNIVERSE}

    # Strong bearish confirmation rotates, rather than de-levers, into tradable defenses.
    if "SPX" in data:
        c = data["SPX"][0]
        bear = c[-1] < c[-6] and c[-1] < c[-21]
        confirmed = bear and c[-1] < c[-31]
        if bear:
            bump = .16 if not confirmed else .22
            for s in ("XAU", "US10Y", "CN10Y"):
                score[s] += bump
            for s in ("BTC", "ETH", "WTI"):
                score[s] = max(.05, score[s] - .75 * bump)

    avg_iv = float(np.mean(list(invvol.values())))
    raw = {s: max(.02, score[s]) * (.70 + .30 * invvol.get(s, avg_iv) / avg_iv) for s in UNIVERSE}
    weights = project_weights(raw)
    vals = np.asarray([score[s] for s in UNIVERSE])
    sd = max(float(vals.std()), 1e-9)
    forecast = {s: float(.01 * (score[s] - vals.mean()) / sd) for s in UNIVERSE}
    try:
        ensemble = json.loads((Path(__file__).parent / "factors/factor_ensemble.json").read_text())
        factor_ids = [x["factor_id"] for x in ensemble.get("selected_factors", []) if isinstance(x, dict)]
    except Exception:
        factor_ids = []
    rebalance_to_weights(weights, forecast_returns=forecast, factor_ids=factor_ids[:10], horizon_days=10)
    last_decision = decision_date
