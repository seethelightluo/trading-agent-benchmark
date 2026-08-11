"""Trader validation v2: simulate online strategy on warm-up data with REAL OHLC.

Slices real OHLC frames to each decision date (no future leakage), drives
strategy.py's own _scores/_weights/_regime/_forecasts over 10-day blocks.
"""
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import strategy as S
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

DATE_FILE = "../persistent/date.json"
tds = json.load(open(DATE_FILE))["trading_days"]
ONLINE_IDX = tds.index(S.ONLINE_START)

assets = list(get_account_dict().get("watch_list", []))
frames = {}
for a in assets:
    df = get_stock_daily_data(symbol=a, days=950)
    if df is not None and len(df) >= 60:
        df = df.reset_index(drop=True)
        df["date_s"] = pd.to_datetime(df["date"]).astype(str)
        frames[a] = df
    else:
        print(f"WARN: {a} insufficient data")
print("frames:", {a: len(df) for a, df in frames.items()})

def df_at(a, d):
    sub = frames[a][frames[a]["date_s"] <= d].tail(S.DATA_DAYS)
    if len(sub) < 30:
        return None
    return sub.reset_index(drop=True)

def run(start_idx, end_idx, label):
    eq = 1.0
    w = {a: 1.0 / len(assets) for a in assets}
    rets_hist, turnovers, decisions = [], [], []
    for i in range(start_idx, end_idx, 10):
        d = tds[i]
        f = {a: df_at(a, d) for a in assets}
        scores, used = S._scores(f, assets)
        if used < 5:
            decisions.append((d, "skip", used, 0.0, 1.0))
            continue
        regime = S._regime(f, assets)
        w_new = S._weights(scores, assets, regime)
        turnover = 0.5 * sum(abs(w_new[a] - w[a]) for a in assets)
        turnovers.append(turnover)
        # forward block returns
        j_end = min(i + 10, end_idx)
        pr, cnt = 0.0, 0
        for a in assets:
            s0 = frames[a][frames[a]["date_s"] <= tds[i]]
            s1 = frames[a][frames[a]["date_s"] <= tds[j_end - 1]]
            if len(s0) == 0 or len(s1) == 0:
                continue
            v0 = float(s0["close"].iloc[-1]); v1 = float(s1["close"].iloc[-1])
            if v0 > 0:
                pr += w_new[a] * (v1 / v0 - 1.0)
                cnt += 1
        cost = 0.0003 * turnover
        r_net = pr - cost
        eq *= (1.0 + r_net)
        rets_hist.append(r_net)
        decisions.append((d, regime, used, round(turnover, 3), round(sum(w_new.values()), 6)))
        w = w_new
    if not rets_hist:
        print(f"\n=== {label}: no blocks ==="); return
    rets = np.array(rets_hist)
    n = len(rets)
    ann = (eq ** (252.0 / (10 * n))) - 1.0 if eq > 0 else -1.0
    sharpe = rets.mean() / rets.std() * np.sqrt(252.0 / 10) if rets.std() > 0 else 0.0
    cum = np.cumprod(1 + rets)
    dd = 1.0 - (cum / np.maximum.accumulate(cum)).min() if eq > 0 else 1.0
    win = (rets > 0).mean()
    print(f"\n=== {label}: {tds[start_idx]} -> {tds[min(end_idx-1, len(tds)-1)]} ===")
    print(f"blocks={n}  end_equity={eq:.4f}  ann_ret={ann:.3%}  sharpe={sharpe:.2f}  maxDD(block)={dd:.2%}  win_rate={win:.2%}")
    print(f"avg 10d ret={rets.mean():.3%}  std={rets.std():.3%}  avg turnover(one-way)={np.mean(turnovers):.3f}")
    print("last rebalances (date, regime, factors, turnover, sum_w):")
    for row in decisions[-5:]:
        print("  ", row)

run(ONLINE_IDX - 250, ONLINE_IDX, "Recent 12m warm-up")
run(ONLINE_IDX - 500, ONLINE_IDX, "2y warm-up")
run(ONLINE_IDX - 750, ONLINE_IDX, "3y warm-up")
print("\nDONE")
