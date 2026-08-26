"""miner_3 cycle 2032-11-11: sweep NEW candidate factor families.
Visible history up to 2032-11-10 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
15-instrument cross-asset universe; >=8 valid instruments per date for IC obs.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2032-11-10'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows = {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float)
        highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
    return closes, highs, lows

closes, highs, lows = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
rets = close.pct_change().dropna()
fwd5  = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR/f'{c}.csv', parse_dates=['date'])
    df = df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
    return df
vix = mac('VIX')

def compute_ic(fv, fwd, min_dates=30):
    fv = fv.reindex(fwd.index)
    ics = []; n_ok = 0
    for d in fwd.index:
        f = fv.loc[d]; r = fwd.loc[d]
        m = f.notna() & r.notna()
        if m.sum() >= 8:
            n_ok += 1
            fv_ = f[m].rank().values; rv_ = r[m].rank().values
            if fv_.std() > 0 and rv_.std() > 0:
                ics.append(np.corrcoef(fv_, rv_)[0,1])
    ics = np.array(ics)
    if len(ics) < min_dates:
        return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0,'cov':0.0,'dates_ok':n_ok}
    hit = float((ics>0).mean()); cov = float(fv.notna().mean().mean())
    mu=ics.mean(); sd=ics.std(); icir=mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov,'dates_ok':n_ok}

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv):
    ic = compute_ic(fv, fwd10); ic5 = compute_ic(fv, fwd5); ic20 = compute_ic(fv, fwd20)
    fv = fv.reindex(fwd10.index)
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]IC={ic5['IC']:.4f} [20]IC={ic20['IC']:.4f}", flush=True)
    return ic

report("A rev_5d", (-rets.rolling(5).mean()).reindex(fwd10.index))

vix_z = (vix - vix.rolling(60).mean())/vix.rolling(60).std()
report("B vix_z60_xsec", pd.DataFrame({a: vix_z for a in ASSETS}).reindex(fwd10.index))

er5 = (close.diff(5).abs()).div((close.diff().abs()).rolling(5).sum()).replace([np.inf,-np.inf],np.nan)
report("C eff_ratio_5d", er5.reindex(fwd10.index))

retrace = (close/close.rolling(120).max()).replace([np.inf,-np.inf],np.nan)
report("D retrace_high120", retrace.reindex(fwd10.index))

def load_ohlc(a):
    f = STOCK_DIR/f'{a}.csv'
    if not f.exists(): f = INDEX_DIR/f'{a}.csv'
    df = pd.read_csv(f, parse_dates=['date'])
    return df[df['date']<=VISIBLE_END].sort_values('date').set_index('date')
intra = pd.DataFrame(index=close.index, columns=ASSETS, dtype=float)
for a in ASSETS:
    d = load_ohlc(a).reindex(close.index)
    rng = (d['high']-d['low']).replace(0,np.nan)
    intra[a] = (d['close']-d['open'])/rng
report("E intraday_eff_10d", intra.rolling(10).mean().reindex(fwd10.index))

vol5 = pd.DataFrame({a: rets[a].rolling(5).std() for a in ASSETS}).reindex(fwd10.index)
vol20 = pd.DataFrame({a: rets[a].rolling(20).std() for a in ASSETS}).reindex(fwd10.index)
report("F vol5_vol20_ratio", (vol5/vol20))

mom30 = pd.DataFrame({a: close[a].pct_change(30) for a in ASSETS}).reindex(fwd10.index)
report("G mom_30d", mom30)

skew60 = pd.DataFrame({a: rets[a].rolling(60).skew() for a in ASSETS}).reindex(fwd10.index)
report("H skew_60d", skew60)

dspring = (close/close.rolling(20).min()-1)
report("I rebound_from_low20", dspring.reindex(fwd10.index))

# J cross-asset: BTC 10d return applied to all assets (crypto carry/trend)
c_all = close.copy()
c_all_pct = c_all.pct_change()
btc_mom = c_all_pct['BTC'].rolling(10).mean()
report("J btc_mom10_xsec", pd.DataFrame({a: btc_mom for a in ASSETS}).reindex(fwd10.index))

# K US10Y momentum applied cross (yield trend risk regime)
us10_mom = close['US10Y'].pct_change(10)
report("K us10y_mom10_xsec", pd.DataFrame({a: us10_mom for a in ASSETS}).reindex(fwd10.index))

# L) 3d momentum
mom3 = pd.DataFrame({a: close[a].pct_change(3) for a in ASSETS}).reindex(fwd10.index)
report("L mom_3d", mom3)