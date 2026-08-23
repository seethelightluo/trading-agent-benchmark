"""miner_1 exploration of novel OHLC/risk-structure candidates @ 2033-05-24."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (
    load_close, load_macro, forward_ret, daily_ic, ic_stats, summarize,
    rank_turnover, coverage_stats, library_panel, max_lib_corr,
    IC_GATE, ICIR_GATE, ASSETS, master_calendar,
)

END = "2033-05-24"
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
lib = library_panel(close, macro)

def load_ohlc(end=END):
    cal = master_calendar(end)
    cols = {}
    for a in ASSETS:
        df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").reindex(cal).ffill()
        cols[a] = df
    op = pd.DataFrame({a: cols[a]["open"] for a in ASSETS})
    hi = pd.DataFrame({a: cols[a]["high"] for a in ASSETS})
    lo = pd.DataFrame({a: cols[a]["low"] for a in ASSETS})
    return op, hi, lo

op, hi, lo = load_ohlc()
tr_lib = lambda name: {k: v for k, v in lib.items() if k != name}

candidates = {}

# 1: low-vol scaled 5d momentum (risk-adjusted short-term trend)
vol20 = ret.rolling(20).std()
candidates["vol_scaled_ret_5x20"] = (close/close.shift(5)-1.0)/vol20

# 2: close position within recent range (intra-range trend position)
rng = (close - lo.rolling(20).min())/(hi.rolling(20).max() - lo.rolling(20).min())
candidates["range_pos_20"] = rng

# 3: price z-score vs MA20 (mean reversion / deviation)
zc = (close - close.rolling(20).mean())/close.rolling(20).std()
candidates["zc_20"] = zc

# 4: up-vol to down-vol asymmetry (semivol ratio, defensive)
up = ret.where(ret > 0, 0.0)
dn = ret.where(ret < 0, 0.0)
candidates["semi_vol_ratio_20"] = (up**2).rolling(20).mean()/((dn**2).rolling(20).mean()+1e-12)

# 5: upper shadow pressure (bearish) = (high-close)/(high-low)
candidates["up_shadow_20"] = ((hi-close)/(hi-lo)).rolling(20).mean()

# 6: 5d drawdown from 20d high (reversion to oversold if deep)
candidates["dd_from_high20"] = (close/close.rolling(20).max() - 1.0)

fw = forward_ret(close, 1)
print(f"=== Exploration @ END={END}: dates={close.shape[0]} assets={close.shape[1]} ===")
out = []
for name, pan in candidates.items():
    ic = daily_ic(pan, fw)
    st = ic_stats(ic, 1)
    full = summarize(pan, close)
    turn = rank_turnover(pan)
    cov = coverage_stats(pan, fw)
    gate = (abs(st["ic"]) >= IC_GATE) and (abs(st["icir"]) >= ICIR_GATE)
    best, pairs = max_lib_corr(pan, tr_lib(name))
    out.append(dict(
        name=name, ic=round(st["ic"],6), icir=round(st["icir"],6),
        hit=round(st["hit"],4), n=st["n"],
        ic_h5=round(full[5]["ic"],6), icir_h5=round(full[5]["icir"],6),
        ic_h10=round(full[10]["ic"],6), icir_h10=round(full[10]["icir"],6),
        turn=round(turn,4), covAD=round(cov["coverage_asset_days"],4),
        covD8=round(cov["coverage_dates_ge8"],4),
        max_abs_lib_corr=round(best,4), lib_pairs=pairs, gate=bool(gate)))
    print(f"{name:22s} IC(h1)={st['ic']:+.4f} ICIR(h1)={st['icir']:+.4f} hit={st['hit']:.3f} "
          f"ic5={full[5]['ic']:+.4f} ic10={full[10]['ic']:+.4f} turn={turn:.2f} gate={gate} maxrho={best:.3f}")

json.dump(out, open("scripts/miner1_20330525_explore.json", "w"), indent=1)
print("saved: scripts/miner1_20330525_explore.json")