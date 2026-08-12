"""Terra AC online strategy (aligned 2026-08-12).

Ensemble-driven: factor_ids/weights/directions are loaded from
factor_ensemble.json at import, exactly like the DeepSeek variant. Signals
are computed deterministically from OHLCV with a 1-bar lag (no look-ahead).
Every rebalance_to_weights call passes forecast_returns + factor_ids so the
migration-cost gate (gross_edge > one-way turnover * 3bp) can decide.
"""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    rebalance_to_weights, register_hook)

ONLINE_START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
STATE_FILE = "trader_state.json"
DATA_DAYS = 170
MIN_ROWS = 140
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CRYPTO = {"BTC", "ETH"}
CYCLICAL_COMMOD = {"WTI", "COPPER"}
CAP_W = 0.16
GUARD_CAP = 0.06
COMP_GUARD_CAP = 0.08
CRYPTO_CAP = 0.12
COMMOD_CAP = 0.14
MOM_TOP_RANK = 0.60
MOM_TOP2_RANK = 0.86
TRAP_PENALTY = 0.50


def _load_ensemble():
    with open("factor_ensemble.json") as f:
        ens = json.load(f)
    return [(x["factor_id"], float(x["weight"]), int(x["direction"]))
            for x in ens["selected_factors"]]


FACTORS = _load_ensemble()


def _num(text, key, default):
    try:
        import re
        m = re.search(key + r"[_ ]*(\d+)", text, re.I)
        return int(m.group(1)) if m else default
    except Exception:
        return default


def _factor_series(df, fid):
    """Deterministic 1-bar-lagged signal for one factor id (family proxy)."""
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    r = c.pct_change()
    lr = np.log(c).diff()
    f = fid.lower()
    if "reversal" in f or "_rev_" in f:
        w = _num(f, "3d", 3) if "3d" in f else (_num(f, "1d", 1) if "1d" in f else 5)
        return -c.pct_change(w).shift(1)
    if "compression" in f or "vol" in f and "comp" in f:
        s = _num(f, "10", 10)
        lw = _num(f, "60", 60)
        return -(lr.rolling(s).std() / lr.rolling(lw).std()).shift(1)
    if "breakout" in f or "distance" in f:
        w = 120 if "120" in f else 60
        return ((c / c.rolling(w).max() - 1.0)).shift(1)
    if "consistency" in f or "persistence" in f or "accel" in f:
        return ((c.pct_change(30)) * (r > 0).rolling(30).mean()).shift(1)
    if "breadth" in f or "asymmetry" in f or "quality" in f:
        return ((r > 0).rolling(30).mean() - 0.5).shift(1)
    if "volatility" in f or "volstate" in f or "_vol_" in f or "risk" in f:
        return (-r.rolling(20).std()).shift(1)
    if "beta" in f or "residual" in f:
        bench = c.pct_change().rolling(60).mean()
        return (r - bench).shift(1)
    if "leadlag" in f or "lead" in f:
        return (c.pct_change(5) - c.pct_change(5).rolling(5).mean()).shift(1)
    if "stress" in f:
        return (-(lr.rolling(5).std() / lr.rolling(20).std())).shift(1)
    if "dispersion" in f:
        return (-r.rolling(5).std()).shift(1)
    if "downside" in f:
        neg = r.copy()
        neg[neg > 0] = np.nan
        return (c.pct_change(20) / (neg.rolling(30).std() + 1e-9)).shift(1)
    # momentum / trend / efficiency default
    return c.pct_change(20).shift(1)


def _factor_values(frames, fid):
    out = {}
    for a, df in frames.items():
        if df is None:
            out[a] = None
            continue
        try:
            s = _factor_series(df, fid)
            s = s.replace([np.inf, -np.inf], np.nan)
            v = float(s.iloc[-1])
            out[a] = v if math.isfinite(v) else None
        except Exception:
            out[a] = None
    return out


def _ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and math.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, n - 1)
    return out


def _scores(frames, assets):
    score = {a: 0.0 for a in assets}
    used = 0
    for fid, w, direction in FACTORS:
        vals = _factor_values(frames, fid)
        if sum(1 for v in vals.values() if v is not None) < 8:
            continue
        r = _ranks(vals, assets)
        for a in assets:
            score[a] += w * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
    return score, used


def _today_and_calendar():
    with open(DATE_FILE) as f:
        d = json.load(f)
    return str(d["current_date"]), d.get("trading_days", [])


def _last_proposal_date(tds):
    try:
        with open(STATE_FILE) as f:
            last = json.load(f).get("last_proposal_date")
            if last and last in tds:
                return last
    except Exception:
        pass
    try:
        acc = get_account_dict()
        last = acc.get("last_rebalance_date")
        if last and last in tds:
            return last
    except Exception:
        pass
    return None


def _should_propose(cur, tds):
    if cur < ONLINE_START or cur not in tds:
        return False
    if (tds.index(cur) - tds.index(ONLINE_START)) % 10 == 0:
        return True
    last = _last_proposal_date(tds)
    if last is None:
        return True
    return (tds.index(cur) - tds.index(last)) >= 10


def _persist_proposal(cur):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_proposal_date": cur}, f)
    except Exception:
        pass


def _fetch(assets):
    frames = {}
    for a in assets:
        try:
            df = get_stock_daily_data(symbol=a, days=DATA_DAYS)
            if df is None or len(df) < MIN_ROWS:
                frames[a] = None
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            frames[a] = df
        except Exception:
            frames[a] = None
    return frames


def _below_ma(frames, assets):
    out = set()
    for a in assets:
        df = frames.get(a)
        if df is not None and len(df) >= 25:
            close = float(df["close"].iloc[-1])
            ma20 = float(df["close"].rolling(20).mean().iloc[-1])
            if math.isfinite(ma20) and close < ma20:
                out.add(a)
    return out


def _weights(scores, assets):
    order = sorted(assets, key=lambda a: (scores[a], a))
    n = len(assets)
    raw = {}
    for i, a in enumerate(order):
        r = i / max(1, n - 1)
        raw[a] = 0.02 + 0.10 * r
    tot = sum(raw.values())
    w = {a: max(0.0, x / tot) for a, x in raw.items()}
    for _ in range(80):
        excess = sum(max(0.0, x - CAP_W) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(CAP_W, x) for a, x in w.items()}
        room = [a for a in w if w[a] < CAP_W - 1e-12]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _forecasts(scores, assets):
    vals = [scores[a] for a in assets]
    mean = float(np.mean(vals))
    half = max(1e-9, (max(vals) - min(vals)) / 2.0)
    f = {}
    for a in assets:
        z = (scores[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    return f


@register_hook
def strategy_hook():
    cur, tds = _today_and_calendar()
    if not _should_propose(cur, tds):
        return
    if not FACTORS:
        return
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    frames = _fetch(assets)
    scores, used = _scores(frames, assets)
    if used < 5:
        w = {a: 1.0 / len(assets) for a in assets}
        w[assets[-1]] += 1.0 - sum(w.values())
        rebalance_to_weights(
            w,
            forecast_returns={a: 0.0 for a in assets},
            factor_ids=[fid for fid, _, _ in FACTORS],
            horizon_days=10,
        )
        _persist_proposal(cur)
        return
    below = _below_ma(frames, assets)
    mom_vals = _factor_values(frames, "mom_120d_skip5") if any("mom_120d" in f for f, _, _ in FACTORS) else {}
    for a in assets:
        mv = mom_vals.get(a)
        if a in below and mv is not None and mv < 0:
            scores[a] -= TRAP_PENALTY
    w = _weights(scores, assets)
    f = _forecasts(scores, assets)
    rebalance_to_weights(
        w,
        forecast_returns=f,
        factor_ids=[fid for fid, _, _ in FACTORS],
        horizon_days=10,
    )
    _persist_proposal(cur)


strategy_hook
