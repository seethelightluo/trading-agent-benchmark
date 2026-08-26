"""miner_3 cycle 2032-08-05: sweep NEW candidate factor families.
Visible history up to 2032-08-04 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084.
Warm-up only: no trading, persistence only for passing factors.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2032-08-04'
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
fwd10 = rets.shift(-10).rolling(10).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}")

# macro observation signals
dxy = pd.read_csv(INDEX_DIR/'DXY.csv', parse_dates=['date'])
dxy = dxy[dxy['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
vix = pd.read_csv(INDEX_DIR/'VIX.csv', parse_dates=['date'])
vix = vix[vix['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
usdcny = pd.read_csv(INDEX_DIR/'USDCNY.csv', parse_dates=['date'])
usdcny = usdcny[usdcny['date']<=VISIBLE_END].set_index('date')['close'].astype(float)

def compute_ic(fv, fwd, min_dates=30):
    fv = fv.reindex(ret_idx)
    ics = []
    n_ok = 0
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
    hit = float((ics > 0).mean())
    cov = float(fv.notna().mean().mean())
    mu = ics.mean(); sd = ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov,'dates_ok':n_ok}

def turnover(fv):
    fv=fv.reindex(ret_idx)
    s = np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv, fwd=None):
    fwd = fwd or fwd10
    ic = compute_ic(fv, fwd)
    print(f"{name}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"dates_ok={ic['dates_ok']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f}")
    return ic

# 1) Global equity beta (SPX beta, 60d) - cross-asset sensitivity to global equity
spx = close['SPX']
sp_ret = spx.pct_change().reindex(ret_idx)
beta_spx = pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    j = pd.concat([rets[a].rename('a'), sp_ret.rename('s')], axis=1).dropna()
    if len(j)>=90:
        beta_spx[a] = (j['s'].rolling(90).cov(j['a'])/j['s'].rolling(90).var())
report("A beta_SPX_90", beta_spx)

# 2) Risk-adjusted momentum: 60d return / 60d realized vol (efficiency spread)
mom60 = pd.DataFrame({a: close[a].pct_change(60) for a in ASSETS})
vol60 = pd.DataFrame({a: rets[a].rolling(60