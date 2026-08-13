"""SCREENER cycle script (2031-03-10) v2 - vectorized. Recompute active-factor signals,
measure recent rank IC (h=10) through visible cutoff 2031-03-07, assess regime,
derive quality-IC-tilt weights. No live-account interaction."""
import json, os
import numpy as np
import pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"
VISIBLE = "2031-03-07"

panel = {}
for a in ASSETS:
    df = pd.read_csv(os.path.join(DATA_DIR, a + ".csv"), parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    s = df.set_index("date")["close"].astype(float)
    panel[a] = s
px = pd.DataFrame(panel).sort_index()
px = px[~px.index.duplicated(keep="last")]
print("panel dates:", px.index.min().date(), "->", px.index.max().date(), "n=", len(px))
print("last close (2031-03-07):")
print(px.iloc[-1].round(3).to_string())

ret = px.pct_change()
mkt = ret.mean(axis=1)

def fast_beta(y, x, win=60, min_obs=40):
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1)
    m = df.notna().all(axis=1)
    df = df[m]
    n = df["x"].rolling(win).count()
    sx = df["x"].rolling(win).sum()
    sy = df["y"].rolling(win).sum()
    sxy = (df["x"] * df["y"]).rolling(win).sum()
    sxx = (df["x"] * df["x"]).rolling(win).sum()
    cov = (sxy - sx * sy / n) / (n - 1)
    var = (sxx - sx * sx / n) / (n - 1)
    beta = cov / var
    beta[(n < min_obs) | (var.abs() <= 1e-12)] = np.nan
    return beta.reindex(y.index)

def dn_mkt_beta(win=60, min_obs=40):
    down = mkt.where(mkt < 0)
    out = pd.DataFrame(index=px.index, columns=ASSETS)
    for a in ASSETS:
        out[a] = fast_beta(ret[a], down, win, min_obs)
    return out

def rate_beta(win=60, min_obs=40):
    cn10y = px["CN10Y"].pct_change()
    out = pd.DataFrame(index=px.index, columns=ASSETS)
    for a in ASSETS:
        out[a] = fast_beta(ret[a], cn10y, win, min_obs)
    return out

def vol_adj_mom_accel(fast=20, slow=60, vwin=20):
    mom_f = px / px.shift(fast) - 1.0
    mom_s = px / px.shift(slow) - 1.0
    vol = ret.rolling(vwin).std()
    return (mom_f - mom_s) / vol

factors = {
    "vol_adj_mom_accel_20x60": vol_adj_mom_accel(),
    "dn_mkt_beta_60d": dn_mkt_beta(),
    "rate_beta_cn10y_60d": rate_beta(),
}
dir_map = {"vol_adj_mom_accel_20x60": 1, "dn_mkt_beta_60d": 1, "rate_beta_cn10y_60d": -1}

fwd = px.shift(-10) / px - 1.0

def rank_ic(sig, fwd_ret, min_valid=8, n_max=None):
    ics = []
    for dt in sig.index:
        s = sig.loc[dt]; f = fwd_ret.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() >= min_valid:
            r = np.corrcoef(s[m].rank(), f[m].rank())[0, 1]
            if np.isfinite(r):
                ics.append((dt, r))
    if not ics:
        return None
    ic = pd.Series([r for _, r in ics], index=[d for d, _ in ics])
    if n_max is not None and len(ic) > n_max:
        ic = ic.iloc[-n_max:]
    return ic

print("\n===== Rank IC (h=10), direction-adjusted =====")
for name, sig in factors.items():
    d = dir_map[name]
    ic_full = rank_ic(sig, fwd)
    print(f"--- {name} (dir {d:+d}) ---")
    if ic_full is None:
        print("  no valid IC dates"); continue
    for n in (None, 750, 500, 250, 120):
        ic = rank_ic(sig, fwd, n_max=n)
        if ic is None or len(ic) == 0:
            continue
        adj = ic * d
        label = f"FULL n={len(ic):4d}" if n is None else f"LAST{n:<4d} n={len(ic):4d}"
        print(f"  {label}  IC_adj={adj.mean():+.4f} ICIR_adj={adj.mean()/adj.std():+.3f} hit={((adj>0).mean()):.3f} raw_IC={ic.mean():+.4f}")
    ic_stats = {"full": ic_full}

print("\n===== Regime (as of 2031-03-07) =====")
for a in ASSETS:
    c = px[a].dropna()
    if len(c) < 210:
        print(f"{a:10s} insufficient history ({len(c)})"); continue
    ma200 = c.rolling(200).mean(); ma60 = c.rolling(60).mean()
    cur = c.iloc[-1]; m200 = ma200.iloc[-1]; m60 = ma60.iloc[-1]
    r60 = c.iloc[-1] / c.iloc[-61] - 1
    r20 = c.iloc[-1] / c.iloc[-21] - 1
    print(f"{a:10s} last={cur:12.4f} vs200d={cur/m200-1:+7.2%} vs60d={cur/m60-1:+7.2%} ret60d={r60:+8.2%} ret20d={r20:+8.2%}")

try:
    vix = pd.read_csv(os.path.join(IDX_DIR, "VIX.csv"), parse_dates=["date"])
    vix = vix[vix["date"] <= pd.Timestamp(VISIBLE)].set_index("date").sort_index()
    vcol = "close" if "close" in vix.columns else vix.columns[1]
    v = vix[vcol].dropna()
    print(f"\nVIX last={v.iloc[-1]:.2f} mean20d={v.iloc[-20:].mean():.2f} mean60d={v.iloc[-60:].mean():.2f} min60d={v.iloc[-60:].min():.2f} max60d={v.iloc[-60:].max():.2f}")
except Exception as e:
    print("VIX err:", e)

disp = ret.rolling(20).std().mean(axis=1)
print(f"\nCS dispersion 20d mean: last={disp.iloc[-1]:.4f} 60dmean={disp.iloc[-60:].mean():.4f} 250dmean={disp.iloc[-250:].mean():.4f}")

print("\n===== Quality tilt (q = |IC|*|ICIR| on last-250d adj IC) =====")
q = {}
for name, sig in factors.items():
    ic = rank_ic(sig, fwd, n_max=250)
    if ic is None or len(ic) < 30:
        print(name, "insufficient recent IC"); continue
    adj = ic * dir_map[name]
    icr = adj.mean() / adj.std()
    q[name] = abs(adj.mean()) * abs(icr)
    print(f"{name:28s} IC_adj={adj.mean():+.4f} ICIR_adj={icr:+.3f} q={q[name]:.5f}")
qsum = sum(q.values())
if qsum > 0:
    w = {k: v / qsum for k, v in q.items()}
    print("\nRaw q-normalized weights:", {k: round(v, 4) for k, v in w.items()})
    ww = dict(w)
    for _ in range(10):
        over = [k for k, v in ww.items() if v > 0.50]
        if not over:
            break
        k = over[0]
        excess = ww[k] - 0.50
        ww[k] = 0.50
        others = [kk for kk in ww if kk != k]
        tot = sum(ww[kk] for kk in others)
        if tot > 0:
            for kk in others:
                ww[kk] += excess * ww[kk] / tot
    print("Capped (<=0.50) weights:", {k: round(v, 4) for k, v in ww.items()}, "sum=", round(sum(ww.values()), 4))
    print("\nRESULT_WEIGHTS =", json.dumps({k: round(v, 4) for k, v in ww.items()}))
