"""miner_1 2028-01-21: scan candidate factor families on 15-asset cross-asset panel."""
import pickle
import numpy as np
import pandas as pd

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C, V, M = panel["close"], panel["vol"], panel["macro"]
ret = C.pct_change()

ONLINE_START = pd.Timestamp("2026-07-16")

def make_factor(name, fn):
    """fn(close_df, vol_df, macro_df) -> DataFrame dates x assets."""
    try:
        return fn(C, V, M)
    except Exception as e:
        print(f"[error] {name}: {e}")
        return None

def fwd_ret(days=1):
    return C.shift(-days) / C - 1.0

def ic_metrics(fac, fwd, min_names=8, sub=None):
    dates = fac.index.intersection(fwd.index)
    if sub is not None:
        dates = dates[(dates >= sub[0]) & (dates <= sub[1])]
    ics, obs = [], []
    for dt in dates:
        f = fac.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_names:
            continue
        x = f[common].astype(float).rank()
        y = r[common].astype(float).rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic); obs.append(len(common))
    ics = np.array(ics)
    if len(ics) == 0:
        return dict(n=0, ic=np.nan, icir=np.nan, hit=np.nan, std=np.nan)
    return dict(n=int(len(ics)), ic=float(ics.mean()),
                icir=float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan,
                hit=float((ics > 0).mean()), std=float(ics.std()))

def coverage(fac):
    n_valid = int(fac.notna().sum().sum())
    n_total = int(C.notna().sum().sum())
    return n_valid / n_total if n_total else np.nan

# ---------- candidate definitions ----------
def f_vscaled_mom(C, V, M):
    mom = C.shift(5) / C.shift(125) - 1.0
    vol = ret.rolling(60).std()
    return mom / vol

def f_mom_ma20_conf(C, V, M):
    mom = C.shift(5) / C.shift(125) - 1.0
    ma20 = C.rolling(20).mean()
    return mom * (C > ma20)

def f_ma20_dist(C, V, M):
    ma20 = C.rolling(20).mean()
    return C / ma20 - 1.0

def f_dd_dist_60(C, V, M):
    return 1.0 - C / C.rolling(60).max()

def f_neg_skew_60(C, V, M):
    return -ret.rolling(60).skew()

def f_vol_ratio_60_20(C, V, M):
    return ret.rolling(60).std() / ret.rolling(20).std()

def f_macd_hist(C, V, M):
    e12 = C.ewm(span=12, adjust=False).mean()
    e26 = C.ewm(span=26, adjust=False).mean()
    return (e12 - e26) / C

def f_idio_mom(C, V, M):
    mom = C.shift(5) / C.shift(125) - 1.0
    xm = mom.mean(axis=1)
    return mom.sub(xm, axis=0)

def f_vol_conf_mom(C, V, M):
    mom = C.shift(5) / C.shift(125) - 1.0
    volz = V.rolling(20).mean()
    return mom * volz.div(volz.mean(axis=1), axis=0)

def f_ma60_slope(C, V, M):
    ma60 = C.rolling(60).mean()
    return ma60 / ma60.shift(20) - 1.0

def f_ucs(C, V, M):
    """up-down capture skew: (mean of up days / mean of down days) over 60d."""
    r = ret.rolling(60).apply(lambda x: (x[x > 0].mean() if (x > 0).any() else 0) /
                              abs(x[x < 0].mean() if (x < 0).any() else np.nan), raw=True)
    return r

CANDIDATES = {
    "vscaled_mom_120x60": f_vscaled_mom,
    "mom_ma20_conf": f_mom_ma20_conf,
    "ma20_dist": f_ma20_dist,
    "dd_dist_60": f_dd_dist_60,
    "neg_skew_60": f_neg_skew_60,
    "vol_ratio_60_20": f_vol_ratio_60_20,
    "macd_hist": f_macd_hist,
    "idio_mom_120": f_idio_mom,
    "vol_conf_mom": f_vol_conf_mom,
    "ma60_slope_20": f_ma60_slope,
    "updown_capture_60": f_ucs,
}

fwd1 = fwd_ret(1)
fwd5 = fwd_ret(5)
fwd10 = fwd_ret(10)

print(f"{'factor':<20} {'cov':>5} {'IC1':>8} {'ICIR1':>7} {'hit1':>5} {'n1':>5} {'IC5':>8} {'IC10':>8} | {'IC1on':>8} {'ICIR1on':>7} {'n1on':>5}")
print("-" * 110)
results = {}
for name, fn in CANDIDATES.items():
    fac = make_factor(name, fn)
    if fac is None:
        continue
    cov = coverage(fac)
    m1 = ic_metrics(fac, fwd1)
    m5 = ic_metrics(fac, fwd5)
    m10 = ic_metrics(fac, fwd10)
    m1on = ic_metrics(fac, fwd1, sub=(ONLINE_START, C.index.max()))
    results[name] = dict(cov=cov, m1=m1, m5=m5, m10=m10, m1on=m1on)
    print(f"{name:<20} {cov:>5.3f} {m1['ic']:>+8.4f} {m1['icir']:>+7.3f} {m1['hit']:>5.3f} {m1['n']:>5d} "
          f"{m5['ic']:>+8.4f} {m10['ic']:>+8.4f} | {m1on['ic']:>+8.4f} {m1on['icir']:>+7.3f} {m1on['n']:>5d}")

pickle.dump(results, open("scripts/miner1_20280121_scanA_results.pkl", "wb"))
print("\nsaved scanA results.")
