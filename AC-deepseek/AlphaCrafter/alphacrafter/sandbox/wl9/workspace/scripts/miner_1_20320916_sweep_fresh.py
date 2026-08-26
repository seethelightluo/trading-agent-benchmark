"""miner_1 cycle 2032-09-16: sweep NEW candidate factor families.
Visible history up to 2032-09-15 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
Warm-up only: no trading, persistence only for passing factors.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2032-09-15'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(assets, end):
    closes, highs, lows, vols = {}, {}, {}, {}
    for a in assets:
        f = STOCK_DIR / f'{a}.csv'
        if not f.exists():
            f = INDEX_DIR / f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date'] <= end].sort_values('date').set_index('date')
        closes[a] = df['close'].astype(float)
        highs[a] = df['high'].astype(float)
        lows[a] = df['low'].astype(float)
        vols[a] = df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan, index=df.index)
    return closes, highs, lows, vols

closes, highs, lows, vols = load(ASSETS, VISIBLE_END)
close = pd.DataFrame(closes).dropna()
high = pd.DataFrame(highs).reindex(close.index)
low = pd.DataFrame(lows).reindex(close.index)
vol = pd.DataFrame(vols).reindex(close.index)
rets = close.pct_change().dropna()
ret_idx = rets.index
fwd5  = rets.shift(-5).rolling(5).mean()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def mac(c):
    df = pd.read_csv(INDEX_DIR/f'{c}.csv', parse_dates=['date'])
    df = df[df['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
    return df
dxy = mac('DXY'); vix = mac('VIX'); usdcny = mac('USDCNY')

def compute_ic(fv, fwd, min_dates=30):
    fv = fv.reindex(ret_idx)
    ics = []; n_ok = 0
    for d in ret_idx:
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
    fv=fv.reindex(ret_idx)
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv, fwd=fwd10):
    ic = compute_ic(fv, fwd)
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f}", flush=True)
    return ic

# 1) Intraday location: (close-low)/(high-low) 20d mean - position within session range (intraday absorption)
range20 = (high.rolling(20).max()-low.rolling(20).min())
clv = (close-low)/(high-low).replace(0,np.nan)
report("A clv_mean_20", clv.rolling(20).mean())

# 2) Amihud illiquidity proxy: |ret| / volume, 20d mean (inverse liquidity premium)
amp = (abs(rets)/(vol+1e9)).rolling(20).mean()
report("B amihud_illiq_20", -amp)  # negative -> illiquid underperform? test

# 3) Range contraction before expansion: (high-low)/close ratio z vs its own 60d mean (volatility regime)
rng = (high-low)/close
rng_z = (rng - rng.rolling(60).mean())/rng.rolling(60).std()
report("C range_z_60", -rng_z)  # low range, contraction = squeeze anticipation

# 4) Volume concentration: recent volume vs 60d average (abnormal volume)
vz = vol/vol.rolling(60).mean()
report("D volsurge_ratio", vz.rolling(5).mean())

# 5) Downside semi-deviation: std of negative returns over 20d (asym risk, low is better)
neg = rets.where(rets<0, 0.0)
semi_neg = (neg**2).rolling(20).mean().pow(0.5)
report("E semi_down_neg_20", -semi_neg)

# 6) Upside participation ratio: sum positive returns / sum abs returns (capture quality)
pos = rets.where(rets>0