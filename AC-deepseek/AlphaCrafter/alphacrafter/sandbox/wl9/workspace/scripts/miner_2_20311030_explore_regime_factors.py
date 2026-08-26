"""miner_2 exploration 2031-10-30: regime/risk and cross-sectional dispersion factors.
Validate on full visible history up to 2031-10-29 (previous completed day).
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.0840 (10d horizon).
Universe = 15 tradable cross-asset instruments.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

VISIBLE_END = '2031-10-29'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

closes, highs, lows, vols = {}, {}, {}, {}
for a in ASSETS:
    df = get_stock_daily_data(symbol=a, days=3000)
    df = df[df['date'] <= VISIBLE_END].set_index('date').sort_index()
    closes[a] = df['close'].astype(float)
    highs[a] = df['high'].astype(float)
    lows[a] = df['low'].astype(float)
    vols[a] = df['volume'].astype(float)

close = pd.DataFrame(closes).dropna()
vol = pd.DataFrame(vols).reindex(close.index)
rets = close.pct_change().dropna()
ret_idx = rets.index
fwd = rets.shift(-10).rolling(10).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}")

def compute_ic(fv):
    fv = fv.reindex(ret_idx)
    ics = []
    for d in ret_idx:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            fv_ = f[m].rank().values; rv_ = r[m].rank().values
            if fv_.std() > 0 and rv_.std() > 0:
                ics.append(np.corrcoef(fv_, rv_)[0,1])
    ics = np.array(ics)
    if len(ics) < 20:
        return {'IC': 0.0, 'ICIR': 0.0, 'n': len(ics), 'hit': 0.0, 'cov': 0.0}
    hit = float((ics > 0).mean())
    cov = float(fv.notna().mean().mean())
    return {'IC': float(ics.mean()),
            'ICIR': float(ics.mean()/ics.std()*np.sqrt(len(ics))) if ics.std()>0 else 0.0,
            'n': len(ics), 'hit': hit, 'cov': cov}

def turnover(fv):
    fv = fv.reindex(ret_idx)
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv):
    ic = compute_ic(fv)
    print(f"{name}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f}")

# A: Average pairwise cross-asset correlation of returns (60d) - regime/risk factor
def avg_pw_corr(window):
    out = pd.Series(np.nan, index=ret_idx)
    for i in range(window, len(ret_idx)):
        w = rets.iloc[i-window:i]
        c = w.corr()
        n = len(c)
        vals = []
        for j in range(n):
            for k in range(j+1,n):
                v = c.iloc[j,k]
                if not np.isnan(v):
                    vals.append(v)
        if len(vals)>=20:
            out.iloc[i] = np.mean(vals)
    return out
corr60 = avg_pw_corr(60)
report("A avg_pw_corr_60", pd.DataFrame({a: corr60 for a in ASSETS}))

# B: Return dispersion across the cross-section (std gap) 20d
disp20 = rets.rolling(20).std().mean(axis=1)
report("B xsec_dispersion_20", pd.DataFrame({a: disp20 for a in ASSETS}))

# C: Downside semideviation / total vol ratio (shortfall asymmetry) 60d
def ds_ratio(window):
    out = pd.DataFrame(index=rets.index, columns=ASSETS, dtype=float)
    for a in ASSETS:
        r = rets[a]
        m = r.rolling(window).mean()
        sq = np.where((r-m) < 0, (r-m)**2, 0.0)
        ds = pd.Series(np.sqrt(pd.Series(sq, index=r.index).rolling(window).mean()), index=r.index)
        tv = r.rolling(window).std()
        out[a] = ds / tv.replace(0, np.nan)
    return out
report("C downside_sdv_ratio_60", ds_ratio(60))

# D: Cross-sectional rank of 60d vol (volatility rebalancing / low-vol tilt)
vol60 = rets.rolling(60).std()
report("D rank_vol_60", vol60)

# E: Log of 10d vs 60d vol ratio (vol term-structure)
vol10 = rets.rolling(10).std()
vts = np.log(vol10 / vol60.replace(0,np.nan))
report("E log_vol_ts_10_60", vts)
