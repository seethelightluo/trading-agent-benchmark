"""miner_2 cycle 7: NOVEL factor families - range-based vol, overnight/intraday
decomposition, volume flow (OBV/MFI/CMF), directional trend, gap activity,
and long-horizon vol-scaled reversal.

Universe: 15 tradable cross-asset instruments (window 2020-01-01..2026-07-15).
Admission gates (benchmark contract): |IC|>=0.007 and |ICIR|>=0.084 @ h=10,
max_abs_library_correlation < 0.5 (self-reported provenance only).
"""
from __future__ import annotations
import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_1_fastlib as lib

EPS = 1e-12
t0 = time.time()

panel = lib.load_panel()
rets = panel.pct_change()
ohlc = lib.load_ohlc_volume()  # per-asset calendar DataFrames
print(f"panel {panel.index[0].date()}..{panel.index[-1].date()} assets={panel.shape[1]} rows={len(panel)} ({time.time()-t0:.1f}s)", flush=True)

# forward returns aligned to factor window (FACTOR_LAST)
fwd_full = {h: lib.fwd_returns(panel, h) for h in lib.HORIZONS}
fwd = {h: fwd_full[h].loc[:lib.FACTOR_LAST] for h in lib.HORIZONS}
fwd_rank_cache = {h: fwd[h].rank(axis=1).values.astype(float) for h in lib.HORIZONS}
libs = lib.library_signals(panel)
print(f"library factors recomputed: {len(libs)} ({time.time()-t0:.1f}s)", flush=True)


def build_union(cols: dict[str, pd.Series]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=panel.index)


# ---------------- per-asset OHLC factor construction ----------------
def per_asset(fn, name):
    out = {}
    for a in lib.WATCH:
        try:
            out[a] = fn(ohlc[a])
        except Exception as e:
            print(f"  {name}/{a} ERR {type(e).__name__}: {e}", flush=True)
    return build_union(out)


def f_parkinson_ratio(c):
    h, l, cl = c["high"].dropna(), c["low"].dropna(), c["close"].dropna()
    idx = h.index.intersection(l.index).intersection(cl.index)
    park = np.sqrt((np.log(h[idx] / l[idx]) ** 2).rolling(20, min_periods=10).mean() / (4 * np.log(2)))
    ccv = cl[idx].pct_change().rolling(20, min_periods=10).std()
    return (park / (ccv + EPS) - 1.0).reindex(cl.index)


def f_gk_ratio(c):
    h, l, o, cl = c["high"].dropna(), c["low"].dropna(), c["open"].dropna(), c["close"].dropna()
    idx = h.index.intersection(l.index).intersection(o.index).intersection(cl.index)
    hl = np.log(h[idx] / l[idx])
    co = np.log(cl[idx] / o[idx])
    gk = np.sqrt(0.5 * (hl ** 2).rolling(20, min_periods=10).mean()
                 - (2 * np.log(2) - 1) * (co ** 2).rolling(20, min_periods=10).mean())
    ccv = cl[idx].pct_change().rolling(20, min_periods=10).std()
    return (gk / (ccv + EPS) - 1.0).reindex(cl.index)


def f_overnight_ret(c):
    o, cl = c["open"].dropna(), c["close"].dropna()
    ovn = o / cl.shift(1) - 1.0
    return ovn.rolling(20, min_periods=10).sum()


def f_intraday_ret(c):
    o, cl = c["open"].dropna(), c["close"].dropna()
    intr = cl / o - 1.0
    return intr.rolling(20, min_periods=10).sum()


def f_overnight_share(c):
    o, cl = c["open"].dropna(), c["close"].dropna()
    ovn = (o / cl.shift(1) - 1.0).rolling(20, min_periods=10).sum().abs()
    intr = (cl / o - 1.0).rolling(20, min_periods=10).sum().abs()
    return ovn / (ovn + intr + EPS)


def f_obv_flow(c):
    cl, v = c["close"].dropna(), c["volume"].dropna()
    idx = cl.index.intersection(v.index)
    sgn = np.sign(cl[idx].diff()).fillna(0.0)
    signed = (sgn * v[idx]).rolling(20, min_periods=10).sum()
    tot = v[idx].rolling(20, min_periods=10).sum()
    return (signed / (tot + EPS)).reindex(cl.index)


def f_mfi(c):
    h, l, cl, v = c["high"].dropna(), c["low"].dropna(), c["close"].dropna(), c["volume"].dropna()
    idx = h.index.intersection(l.index).intersection(cl.index).intersection(v.index)
    tp = (h[idx] + l[idx] + cl[idx]) / 3.0
    rmf = tp * v[idx]
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(14, min_periods=7).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(14, min_periods=7).sum()
    mfi = 100.0 - 100.0 / (1.0 + pos / (neg + EPS))
    return (mfi / 100.0).reindex(cl.index)


def f_cmf(c):
    h, l, cl, v = c["high"].dropna(), c["low"].dropna(), c["close"].dropna(), c["volume"].dropna()
    idx = h.index.intersection(l.index).intersection(cl.index).intersection(v.index)
    mfm = ((cl[idx] - l[idx]) - (h[idx] - cl[idx])) / (h[idx] - l[idx] + EPS)
    num = (mfm * v[idx]).rolling(20, min_periods=10).sum()
    den = v[idx].rolling(20, min_periods=10).sum()
    return (num / (den + EPS)).reindex(cl.index)


def f_adx(c, win=14):
    h, l, cl = c["high"].dropna(), c["low"].dropna(), c["close"].dropna()
    idx = h.index.intersection(l.index).intersection(cl.index)
    hh, ll, pc = h[idx], l[idx], cl[idx].shift(1)
    up = hh - hh.shift(1)
    dn = ll.shift(1) - ll
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=idx)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=idx)
    tr = pd.concat([hh - ll, (hh - pc).abs(), (ll - pc).abs()], axis=1).max(axis=1)
    alpha = 1.0 / win
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=win).mean()
    pdi = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=win).mean() / (atr + EPS)
    mdi = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=win).mean() / (atr + EPS)
    dx = 100.0 * (pdi - mdi).abs() / (pdi + mdi + EPS)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=win).mean()
    return (adx / 100.0).reindex(cl.index)


def f_di_ratio(c, win=14):
    h, l, cl = c["high"].dropna(), c["low"].dropna(), c["close"].dropna()
    idx = h.index.intersection(l.index).intersection(cl.index)
    hh, ll, pc = h[idx], l[idx], cl[idx].shift(1)
    up = hh - hh.shift(1)
    dn = ll.shift(1) - ll
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=idx)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=idx)
    tr = pd.concat([hh - ll, (hh - pc).abs(), (ll - pc).abs()], axis=1).max(axis=1)
    alpha = 1.0 / win
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=win).mean()
    pdi = 100.0 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=win).mean() / (atr + EPS)
    mdi = 100.0 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=win).mean() / (atr + EPS)
    return ((pdi - mdi) / (pdi + mdi + EPS)).reindex(cl.index)


def f_gap_vol(c):
    o, cl = c["open"].dropna(), c["close"].dropna()
    gap = (o / cl.shift(1) - 1.0).abs()
    ccv = cl.pct_change().rolling(20, min_periods=10).std()
    return (gap.rolling(20, min_periods=10).mean() / (ccv + EPS)).reindex(cl.index)


def f_gap_sign(c):
    o, cl = c["open"].dropna(), c["close"].dropna()
    gap = o / cl.shift(1) - 1.0
    return gap.rolling(20, min_periods=10).mean()


# ---------------- panel-level candidates ----------------
v20 = rets.rolling(20, min_periods=10).std()
v60 = rets.rolling(60, min_periods=30).std()
m20 = panel.shift(5) / panel.shift(25) - 1.0
m60 = panel.shift(5) / panel.shift(65) - 1.0

C = {}
C["parkinson_ratio_20"] = per_asset(f_parkinson_ratio, "parkinson")
C["gk_ratio_20"] = per_asset(f_gk_ratio, "gk")
C["overnight_ret_20"] = per_asset(f_overnight_ret, "ovn")
C["intraday_ret_20"] = per_asset(f_intraday_ret, "intr")
C["overnight_share_20"] = per_asset(f_overnight_share, "ovnshare")
C["obv_flow_20"] = per_asset(f_obv_flow, "obv")
C["mfi_14"] = per_asset(f_mfi, "mfi")
C["cmf_20"] = per_asset(f_cmf, "cmf")
C["adx_14"] = per_asset(f_adx, "adx")
C["di_ratio_14"] = per_asset(f_di_ratio, "di")
C["gap_vol_20"] = per_asset(f_gap_vol, "gapvol")
C["gap_sign_20"] = per_asset(f_gap_sign, "gapsign")
C["rev_20d_vol"] = -(m20 / (v20 + EPS))
C["ts_mom_60_vol_adj"] = m60 / (v60 + EPS)

print(f"candidates built ({time.time()-t0:.1f}s)", flush=True)

RESULTS = {}
for name, f in C.items():
    try:
        RESULTS[name] = lib.validate_fast(name, f, panel, fwd, libs, fwd_rank_cache)
    except Exception as e:
        print(f"=== {name}: ERROR {type(e).__name__}: {e} ===", flush=True)

json.dump(RESULTS, open("scripts/miner_2_cycle7_results.json", "w"), indent=1, default=str)
print("\nSAVED scripts/miner_2_cycle7_results.json", flush=True)
print("\n===== SUMMARY =====", flush=True)
for name, r in RESULTS.items():
    if r is None:
        continue
    p = r["admission_gate"]["pass"]
    print(f"{name:<22} IC={r['ic_h10']:+.4f} ICIR={r['icir_h10']:+.4f} "
          f"hit={r['hit_h10']:.3f} cov={r['coverage_asset_days']:.3f} "
          f"turn={r['turnover_10d_rank']:.2f} maxcorr={r['max_abs_library_correlation']} "
          f"-> {'PASS' if p else 'FAIL'}", flush=True)
