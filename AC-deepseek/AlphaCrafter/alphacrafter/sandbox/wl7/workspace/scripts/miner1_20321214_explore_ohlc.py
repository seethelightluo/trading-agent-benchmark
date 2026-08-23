"""miner_1 candidate exploration at 2032-12-14: proper OHLC candle-location factor.
Uses actual intraday high/low from CSV to test close-location-in-range and candle body
bias, which are genuinely orthogonal to the close-return library.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import miner_shared as ms

DATA_DIR = ms.DATA_DIR

def load_ohlc(end="2032-12-13"):
    cal = ms.master_calendar(end)
    o = pd.DataFrame(index=cal); h = pd.DataFrame(index=cal)
    l = pd.DataFrame(index=cal); c = pd.DataFrame(index=cal)
    for a in ms.ASSETS:
        df = pd.read_csv(f"{DATA_DIR}/{a}.csv")
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")
        o[a] = s["open"].reindex(cal).ffill()
        h[a] = s["high"].reindex(cal).ffill()
        l[a] = s["low"].reindex(cal).ffill()
        c[a] = s["close"].reindex(cal).ffill()
    return o, h, l, c

END = "2032-12-13"
o, hi, lo, cl = load_ohlc(END)
macro = ms.load_macro(END)
lib = ms.library_panel(cl, macro)
fwd = ms.forward_ret(cl, 10)

def close_location(o, hi, lo, cl, window=20, skip=5):
    rng = hi.rolling(window).max() - lo.rolling(window).min()
    k = (cl - lo.rolling(window).min()) / rng.replace(0, np.nan)
    return (k.shift(skip) - 0.5)

def candle_body(o, hi, lo, cl, skip=5):
    """(close-open)/max(hi-lo,eps) - signed candle body strength."""
    rng = (hi - lo).replace(0, np.nan)
    body = (cl - o) / rng
    return body.shift(skip)

def down_colon_up_ratio(o, hi, lo, cl, window=10, skip=5):
    # fraction of red (down) candles over window, mean-centered
    body = (cl - o) / (hi - lo).replace(0, np.nan)
    neg = (body < 0).rolling(window).mean()
    return (neg.shift(skip) - 0.5)

cands = {
    "close_loci_20x5": close_location(o, hi, lo, cl, 20, 5),
    "candle_body_5": candle_body(o, hi, lo, cl, 5),
    "down_frac_10x5": down_colon_up_ratio(o, hi, lo, cl, 10, 5),
}

rows_out = []
for name, f in cands.items():
    st = ms.ic_stats(ms.daily_ic(f, fwd), 10)
    ic_r = ms.ic_stats(ms.daily_ic(f.tail(500), ms.forward_ret(cl, 10).reindex(f.tail(500).index)), 10)
    ic_q = ms.ic_stats(ms.daily_ic(f.tail(250), ms.forward_ret(cl, 10).reindex(f.tail(250).index)), 10)
    cov = ms.coverage_stats(f, fwd)
    turn = ms.rank_turnover(f, 10)
    best, pairs = ms.max_lib_corr(f.tail(500), lib)
    dec = {h: ms.ic_stats(ms.daily_ic(f, ms.forward_ret(cl, h)), h)["ic"] for h in (3, 5, 10, 20)}
    gate = abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE
    rows_out.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                         ic_r=ic_r["ic"], icir_r=ic_r["icir"], n_r=ic_r["n"],
                         ic_q=ic_q["ic"], icir_q=ic_q["icir"], n_q=ic_q["n"],
                         covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"],
                         turn=turn, max_abs_lib_corr=round(best, 4), decay=dec, gate=bool(gate)))

print("candidate                          IC10     ICIR10   hit    n | IC_r  ICIR_r | IC_q  ICIR_q | covAD  covD8  turn  maxRho  decay{3,5,10,20}  gate")
for r in rows_out:
    print(f"{r['name']:28s} {r['ic']:+.4f} {r['icir']:+.3f} {r['hit']:.3f} {r['n']:5d} | "
          f"{r['ic_r']:+.4f} {r['icir_r']:+.3f} | {r['ic_q']:+.4f} {r['icir_q']:+.3f} | "
          f"{r['covAD']:.3f} {r['covD8']:.3f} {r['turn']:.2f} {r['max_abs_lib_corr']:.3f}  "
          f"{r['decay']}  {'PASS' if r['gate'] else 'FAIL'}")

json.dump(rows_out, open("scripts/miner1_20321214_explore_ohlc.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner1_20321214_explore_ohlc.json")