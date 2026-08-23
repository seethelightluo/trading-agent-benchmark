"""Exploration 2026-10-08: intraday close-position / body-strength factor family.

Idea: measure where each asset's close sits within its recent daily candlestick
ranges across a lookback. Persistent intraday placement (e.g., closing strength
in the upper part of the daily range) may encode demand/supply imbalance that
predicts forward returns cross-sectionally.

Variants:
  - body_pos_Nd  : mean of body ratio (close-open)/range over N days
  - upbody_ratio_Nd : fraction of days where close is in upper half of range
  - close_low_prox_Nd : mean (close - low)/(high - low)

Mixed calendars (BTC/ETH 7d/wk) handled per-asset before reindexing.
Admission gate (15-asset universe): abs daily paper IC >= 0.0070 and
abs daily paper ICIR >= 0.0840 at horizon h=10, min_valid=8.
Window: 2020-01-01 .. 2026-10-07 (visible through previous completed trading day).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_1_20260730_helpers import (WATCH, load_panel, forward_returns,
                                      factor_ic_report, factor_turnover,
                                      coverage, decay_report,
                                      max_library_correlation)

closes = load_panel(WATCH)
rets = closes.pct_change()
print(f"closes shape: {closes.shape}  range: {closes.index.min().date()} -> {closes.index.max().date()}")


def load_ohlc(symbol, root="../persistent"):
    for sub, name in (("stock_data", symbol),):
        p = f"{root}/{sub}/{name}.csv"
        try:
            df = pd.read_csv(p, parse_dates=["date"])
        except Exception:
            continue
        df = df[df["date"] <= "2026-10-07"].set_index("date")
        return df
    return None


def intraday_factor(fn, lookbacks=(5, 10, 20)):
    out = {}
    for a in WATCH:
        df = load_ohlc(a)
        if df is None or len(df) < lookbacks[-1] + 5:
            out[a] = pd.Series(np.nan, index=closes.index)
            continue
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = (h - l).replace(0, np.nan)
        body = (c - o) / rng
        upp = ((c - l) / rng) > 0.5
        upb = ((c - o) / rng) > 0.0
        for look in lookbacks:
            f = fn(body, upp, upb, c, l, h, look)
            out[f"{a}__{look}"] = f.reindex(closes.index)
    return pd.concat(out, axis=1) if False else pd.DataFrame(out)


def run_var(name, panel):
    h = 10
    fwd = forward_returns(rets, h)
    rep = factor_ic_report(panel, fwd, horizon=h)
    if rep is None:
        print(f"{name:<28} insufficient data")
        return None
    turn = factor_turnover(panel)
    cov = coverage(panel)
    dec = decay_report(panel, rets)
    passed = abs(rep["ic"]) >= 0.0070 and abs(rep["icir"]) >= 0.0840
    print(f"{name:<28} IC={rep['ic']:>8.4f} ICIR={rep['icir']:>8.4f} "
          f"hit={rep['ic_hit_ratio']:>5.3f} n={rep['n_ic_dates']:>5d} "
          f"meanN={rep['mean_n_valid']:>4.1f} turn={turn:>5.2f} "
          f"cov_date8={cov['coverage_dates_ge8']:>4.2f} decay1/5/10/20= "
          f"{dec['1']}/{dec['5']}/{dec['10']}/{dec['20']}  PASS={passed}")
    return rep


# Build per-lookback panels indexed by union axis with columns = assets
def assemble(fn, look):
    cols = {}
    for a in WATCH:
        df = load_ohlc(a)
        if df is None or len(df) < 30:
            continue
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = (h - l).replace(0, np.nan)
        body = (c - o) / rng
        upp = ((c - l) / rng) > 0.5
        upb = ((c - o) / rng) > 0.0
        cols[a] = fn(body, upp, upb, c, l, h, look).reindex(closes.index)
    return pd.DataFrame(cols)


def a_body(body, upp, upb, c, l, h, look):
    return body.rolling(look).mean()
def a_upbody(body, upp, upb, c, l, h, look):
    return upb.rolling(look).mean()
def a_close_low(body, upp, upb, c, l, h, look):
    rng = (h - l).replace(0, np.nan)
    return ((c - l) / rng).rolling(look).mean()

print("\n== Intraday body/position family (validation window to 2026-10-07) ==\n")
results = {}
for look in (5, 10, 20):
    results[f"body_pos_{look}d"] = run_var(f"body_pos_{look}d", assemble(a_body, look))
    results[f"upbody_ratio_{look}d"] = run_var(f"upbody_ratio_{look}d", assemble(a_upbody, look))
    results[f"close_low_prox_{look}d"] = run_var(f"close_low_prox_{look}d", assemble(a_close_low, look))