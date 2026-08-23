"""miner_1 candidate exploration at 2032-12-14: candle-range pressure (bullish close-location).
Tests whether assets that close near their 20d high/low (intraday range memory) were
overlooked by the active library (which is return-based: rel_mom, beta_ew, max_ret, kurt,
downside_vol, corr). Also tests a volume-shock variant and decay by horizon.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner_shared as ms

END = "2032-12-13"
close = ms.load_close(END)
macro = ms.load_macro(END)

def candle_range_pressure(close, window=20, skip=5):
    """(close - min(low,window)) / (max(high,window)-min(low,window)) - 0.5
    Ranges [0,1] - mean-centered. Favors assets closing near intraday highs.
    Implemented with rolling on series."""
    # need high/low - load raw closes approximate via (high/low not in panel)
    # Instead approximate range with close-based proxy: use close min/max (Stochastic K proxy).
    rng = close.rolling(window).max() - close.rolling(window).min()
    k = (close - close.rolling(window).min()) / rng.replace(0, np.nan)
    return (k.shift(skip) - 0.5)

def vol_shock(close, av_window=20, break_window=5, skip=5):
    """(ret_5d vol / ret_20d vol - 1) shifted - volatility regime breakout signal."""
    ret = close.pct_change()
    short = ret.rolling(break_window).std()
    long = ret.rolling(av_window).std()
    return (short / long - 1.0).shift(skip)

lib = ms.library_panel(close, macro)
fwd = ms.forward_ret(close, 10)

cands = {
    "range_pressure_20x5": candle_range_pressure(close, 20, 5),
    "ret_vol_ratio_5x20": vol_shock(close, 20, 5, 5),
}

# compute gate + lib corr and multi-horizon decay
rows_out = []
for name, f in cands.items():
    st = ms.ic_stats(ms.daily_ic(f, fwd), 10)
    ic_r = ms.ic_stats(ms.daily_ic(f.tail(500), ms.forward_ret(close, 10).reindex(f.tail(500).index)), 10)
    ic_q = ms.ic_stats(ms.daily_ic(f.tail(250), ms.forward_ret(close, 10).reindex(f.tail(250).index)), 10)
    cov = ms.coverage_stats(f, fwd)
    turn = ms.rank_turnover(f, 10)
    best, pairs = ms.max_lib_corr(f.tail(500), lib)
    # decay across horizons
    dec = {}
    for h in (3, 5, 10, 20):
        dec[h] = ms.ic_stats(ms.daily_ic(f, ms.forward_ret(close, h)), h)["ic"]
    gate = abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE
    rows_out.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                         ic_r=ic_r["ic"], icir_r=ic_r["icir"], n_r=ic_r["n"],
                         ic_q=ic_q["ic"], icir_q=ic_q["icir"], n_q=ic_q["n"],
                         covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"],
                         turn=turn, max_abs_lib_corr=round(best, 4), decay=dec,
                         lib_pairs=pairs, gate=bool(gate)))

print("candidate                          IC10     ICIR10   hit    n | IC_r  ICIR_r | IC_q  ICIR_q | covAD  covD8  turn  maxRho  decay{3,5,10,20}  gate")
for r in rows_out:
    print(f"{r['name']:28s} {r['ic']:+.4f} {r['icir']:+.3f} {r['hit']:.3f} {r['n']:5d} | "
          f"{r['ic_r']:+.4f} {r['icir_r']:+.3f} | {r['ic_q']:+.4f} {r['icir_q']:+.3f} | "
          f"{r['covAD']:.3f} {r['covD8']:.3f} {r['turn']:.2f} {r['max_abs_lib_corr']:.3f}  "
          f"{r['decay']}  {'PASS' if r['gate'] else 'FAIL'}")
    if r["gate"]:
        print("   lib correlations:", r["lib_pairs"])

json.dump(rows_out, open("scripts/miner1_20321214_explore.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner1_20321214_explore.json")