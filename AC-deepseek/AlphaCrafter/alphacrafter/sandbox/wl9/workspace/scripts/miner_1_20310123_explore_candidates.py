#!/usr/bin/env python
"""Miner_1 exploration: screen candidate factor families on the 15-asset cross-section.
Current date 2031-01-23, visible through 2031-01-22. Validation uses data up to visible date only.
"""
import pandas as pd, numpy as np

CUR = "2031-01-22"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data/"
IDATA = "../persistent/index_data/"

def load_series(assets, val="close"):
    out = {}
    for a in assets:
        df = pd.read_csv(DATA+a+".csv")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df[df["date"] <= CUR]
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        out[a] = df[val]
    return pd.DataFrame(out).sort_index()

px = load_series(ASSETS, "close")
hi = load_series(ASSETS, "high")
lo = load_series(ASSETS, "low")
print("Panel:", px.shape, px.index[0].date(), "..", px.index[-1].date())

def load_macro(name, col="close"):
    df = pd.read_csv(IDATA+name+".csv")
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df[df["date"] <= CUR]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[col]
vix = load_macro("VIX")

r = px.pct_change()
def forward_ret(h):
    return px.shift(-h) / px - 1.0

def daily_ic(signal, fwd, min_instr=8):
    ics = []
    for dt in fwd.index:
        if dt not in signal.index: continue
        s = signal.loc[dt]; f = fwd.loc[dt]
        m = s.notna() & f.notna() & np.isfinite(s) & np.isfinite(f)
        if m.sum() >= min_instr:
            try:
                ic = np.corrcoef(s[m], f[m])[0,1]
                if np.isfinite(ic): ics.append(ic)
            except Exception: pass
    ics = np.array(ics)
    if len(ics)==0: return dict(ic=np.nan, icir=np.nan, hit=np.nan, ndates=0)
    ic = ics.mean(); ic_std = ics.std()
    return dict(ic=ic, icir=ic/ic_std if ic_std>0 else 0.0, hit=(ics>0).mean(), ndates=len(ics))

cand = {}
cand["mom10_skip2"] = px.shift(2)/px.shift(12)-1.0
cand["mom20_skip10"] = px.shift(10)/px.shift(30)-1.0
cand["mom60_skip20"] = px.shift(20)/px.shift(80)-1.0
mom10 = px.shift(5)/px.shift(15)-1.0
cand["mom_accel10"] = mom10 - mom10.shift(10)
cand["range_hl_20"] = (hi.rolling(20).max() - lo.rolling(20).min())/px
cand["rng_pos_hl_20"] = (px - lo.rolling(20).min())/(hi.rolling(20).max()-lo.rolling(20).min())
cand["skew_10d"] = r.rolling(10).skew()
cand["kurt_10d"] = r.rolling(10).kurt()
cand["intraday_range_20"] = ((hi-lo)/px).rolling(20).mean()
def downside_ratio(x):
    neg = x[x<0]
    if len(neg)==0: return 0.0
    return -neg.mean()/(x.std()+1e-12)
cand["downside_ratio_20"] = r.rolling(20).apply(downside_ratio, raw=True)
def cap_ratio(x):
    pos = x[x>0].sum(); neg = -x[x<0].sum()
    return pos/neg if neg>0 else 0.0
cand["capture_ratio_20"] = r.rolling(20).apply(cap_ratio, raw=True)
def ac5(x):
    if len(x)<8: return np.nan
    return np.corrcoef(x[:-1], x[1:])[0,1]
cand["ac5_20d"] = r.rolling(20).apply(ac5, raw=True)
cand["dist_ma250"] = px/px.rolling(250).mean()-1.0
cand["vol_std_120"] = (px-px.rolling(120).mean())/px.rolling(120).mean()
cand["mom_breakout_20"] = px.shift(5)/px.shift(25)-1.0
def beta_vix_series(asset_col, win=20):
    j = pd.concat([asset_col.pct_change(), vix.pct_change()], axis=1, keys=['a','v']).dropna()
    cov = j['a'].rolling(win).cov(j['v'])
    var = j['v'].rolling(win).var()
    return (cov/var).reindex(px.index)
cand["beta_vix_20"] = px.apply(lambda c: beta_vix_series(c,20))

horizons = [5,10,20]
print("\n=== Candidate screening (h=10 admission horizon) ===")
print(f"{'name':<22}{'ic':>8}{'icir':>8}{'hit':>7}{'ndates':>8}")
results_all = {}
for name, sig in cand.items():
    fwd = forward_ret(10)
    res = daily_ic(sig, fwd)
    results_all[name] = res
    if np.isfinite(res['ic']):
        print(f"{name:<22}{res['ic']:>8.4f}{res['icir']:>8.3f}{res['hit']:>7.3f}{res['ndates']:>8}")

print("\n=== Passing candidates (abs ic>=0.018, abs icir>=0.084) decay===")
passing = {k:v for k,v in results_all.items() if np.isfinite(v['ic']) and abs(v['ic'])>=0.018 and abs(v['icir'])>=0.084}
for name in sorted(passing, key=lambda x: -abs(results_all[x]['ic'])):
    print(f"\n[{name}] ic={results_all[name]['ic']:.4f} icir={results_all[name]['icir']:.3f} hit={results_all[name]['hit']:.3f} ndates={results_all[name]['ndates']}")
    for h in horizons:
        rr = daily_ic(cand[name], forward_ret(h))
        if np.isfinite(rr['ic']):
            print(f"  h={h}: ic={rr['ic']:.4f} icir={rr['icir']:.3f}")