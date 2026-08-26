"""miner_3 cycle 2032-09-16: sweep NEW candidate factor families.
Visible history up to 2032-09-15 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084 (10d horizon).
Warm-up only: persistence only for passing factors.
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
    ic = compute_ic(fv, fwd); ic5 = compute_ic(fv, fwd5); ic20 = compute_ic(fv, fwd20)
    print(f"{name}[10]: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f} "
          f"| [5]IC={ic5['IC']:.4f} [20]IC={ic20['IC']:.4f}", flush=True)
    return ic

# A) Pullback depth within 5d (fresh dip reversion)
rh5 = close.rolling(5).max()
prox5 = (close/rh5 - 1).reindex(ret_idx)
report("A depth_from_high5", prox5)

# B) Upside/downside volatility asymmetry
retp = rets.where(rets>0, 0.0); retn = rets.where(rets<0, 0.0)
vol_up = pd.DataFrame({a: retp[a].rolling(20).std() for a in ASSETS}).reindex(ret_idx)
vol_dn = pd.DataFrame({a: abs(retn[a]).rolling(20).std() for a in ASSETS}).reindex(ret_idx)
report("B downup_vol_asym20", (vol_dn - vol_up).reindex(ret_idx))

# C) Range position (close within [low,high]) averaged 5d
rngpos = ((close - low)/(high-low).replace(0,np.nan)).rolling(5).mean().reindex(ret_idx)
report("C range_pos5", rngpos)

# D) WTI/XAU relative momentum (commodity risk appetite)
wti_xau = close['WTI']/close['XAU']
rel_mom = wti_xau.pct_change(20)
D = pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float); D[:] = rel_mom.values[:,None]
report("D wti_xau_relmom20", D)

# E) DXY trend applied cross-sectionally
dt = dxy.pct_change(20).reindex(ret_idx)
E = pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float); E[:] = dt.values[:,None]
report("E dxy_trend20", E)

# F) Recovery from 60d low (position within 60d range)
lo60 = close.rolling(60).min(); hi60 = close.rolling(60).max()
recover = ((close - lo60)/(hi60 - lo60)).replace([np.inf,-np.inf],np.nan).reindex(ret_idx)
report("F recover_from_low60", recover)

# G) 20d realized vol direction (10d-40d vol diff)
vol_short = pd.DataFrame({a: rets[a].rolling(10).std() for a in ASSETS})
vol_long  = pd.DataFrame({a: rets[a].rolling(40).std() for a in ASSETS})
report("G vol_10_40_diff", (vol_short - vol_long).reindex(ret_idx))

# H) 3d ROC volatility-scaled (recent momentum normalized by vol)
mom3 = pd.DataFrame({a: close[a].pct_change(3) for a in ASSETS})
vol10 = pd.DataFrame({a: rets[a].rolling(10).std() for a in ASSETS})
report("H mom3_vol10", (mom3/vol10).reindex(ret_idx))

# I) DXY beta (asset sensitivity to USD basket)
dxr = dxy.pct_change().reindex(ret_idx)
dxy_beta = pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), dxr.rename('d')], axis=1).dropna()
    if len(j)>=90:
        dxy_beta[a] = j['d'].rolling(90).cov(j['a'])/j['d'].rolling(90).var()
report("I dxy_beta90", dxy_beta)

# J) Distance below 10d high (reversal into dips, short lookback)
rh10 = close.rolling(10).max()
report("J depth_from_high10", (close/rh10 - 1).reindex(ret_idx))
