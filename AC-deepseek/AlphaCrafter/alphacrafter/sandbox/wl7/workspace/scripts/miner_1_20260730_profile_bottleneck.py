"""miner_1: profile components of cycle8 script to locate the timeout bottleneck."""
from __future__ import annotations
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_2_lib as lib

EPS = 1e-12
t0 = time.time()

# 1) load panel
t = time.time()
panel = lib.load_panel()
macro = lib.load_macro()
print(f"[load_panel+macro] {time.time()-t:.2f}s (panel {panel.shape})", flush=True)

t = time.time()
rets = panel.pct_change()
mkt_r = panel.mean(axis=1).pct_change()
print(f"[pct_change] {time.time()-t:.2f}s", flush=True)

# 2) volume panel
t = time.time()
vols = {}
for a in lib.WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
    vols[a] = df["volume"].astype(float)
vol_panel = pd.DataFrame(vols, index=panel.index)
print(f"[vol_panel load] {time.time()-t:.2f}s", flush=True)

# 3) candle cols (re-reads CSVs a third time)
t = time.time()
candle_cols = {}
for a in lib.WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{a}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= lib.MAX_VISIBLE].set_index("date").sort_index()
    body = (df["close"] - df["open"]) / (df["high"] - df["low"] + EPS)
    candle_cols[a] = body.rolling(20, min_periods=10).mean()
_ = pd.DataFrame(candle_cols, index=panel.index)
print(f"[candle load+calc] {time.time()-t:.2f}s", flush=True)

# 4) rolling beta: check pandas cov shape/time vs vectorized
def rolling_beta_fast(x: pd.DataFrame, m: pd.Series, win=60, minp=30) -> pd.DataFrame:
    w = float(win)
    m = m.reindex(x.index)
    sm = m.rolling(win, min_periods=minp).sum()
    sx = x.rolling(win, min_periods=minp).sum()
    sxm = (x.mul(m, axis=0)).rolling(win, min_periods=minp).sum()
    smm = (m * m).rolling(win, min_periods=minp).sum()
    cov = (sxm - sx.mul(sm, axis=0) / w) / (w - 1)
    var = (smm - sm * sm / w) / (w - 1)
    return cov.div(var.replace(0, np.nan), axis=0)

t = time.time()
bf = rolling_beta_fast(rets, panel["BTC"].pct_change(), 60, 30)
print(f"[rolling_beta_fast (1 call)] {time.time()-t:.2f}s shape={bf.shape}", flush=True)

t = time.time()
vixr = macro["VIX"].pct_change()
bf2 = rolling_beta_fast(rets, vixr, 60, 30)
print(f"[rolling_beta_fast (VIX)] {time.time()-t:.2f}s", flush=True)

# sanity: beta of BTC vs BTC leader should be ~1
print("   sanity BTC self-beta tail:", np.nanmean(bf["BTC"].tail(100)), flush=True)

# 5) library corr full (per-date spearman loop) - timing with 2 libs
from scipy.stats import spearmanr

def library_corr_full(factor: pd.DataFrame, libs: dict) -> tuple:
    per = {}
    common = factor.index.intersection(panel.index)
    n_calls = 0
    t = time.time()
    for fid, lf in libs.items():
        cs = []
        for dt in common[-500:]:
            f = factor.loc[dt]
            g = lf.loc[dt].reindex(f.index)
            m = f.notna() & g.notna() & np.isfinite(f.astype(float)) & np.isfinite(g.astype(float))
            m = m.reindex(f.index).fillna(False)
            if int(m.sum()) >= lib.MIN_ASSETS:
                cs.append(spearmanr(f[m], g[m])[0])
                n_calls += 1
        per[fid] = round(float(np.mean(cs)), 4) if cs else None
    print(f"   library_corr_full: {time.time()-t:.2f}s for {len(libs)} libs x 500 dates, {n_calls} spearman calls", flush=True)
    valid = [abs(v) for v in per.values() if v is not None]
    return (round(max(valid), 4) if valid else float("nan")), per

libs = {
    "mom_10d_skip5": panel.shift(5) / panel.shift(15) - 1.0,
    "mom_120d_skip5": panel.shift(5) / panel.shift(125) - 1.0,
}
t = time.time()
mc, per = library_corr_full(bf.loc[:lib.FACTOR_LAST], libs)
print(f"[library_corr_full 2 libs] {time.time()-t:.2f}s maxcorr={mc}", flush=True)

# 6) turnover_10d_rank
t = time.time()
tr = lib.turnover_10d_rank(bf)
print(f"[turnover_10d_rank] {time.time()-t:.2f}s -> {tr:.3f}", flush=True)

# 7) fwd returns 6 horizons
t = time.time()
fwd = {h: lib.fwd_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}
print(f"[fwd_returns x6] {time.time()-t:.2f}s", flush=True)

# 8) rank_ic_series on one factor
t = time.time()
ics = lib.rank_ic_series(bf.loc[:lib.FACTOR_LAST], fwd[10])
print(f"[rank_ic_series] {time.time()-t:.2f}s n={len(ics)}", flush=True)

print(f"TOTAL {time.time()-t0:.2f}s", flush=True)
