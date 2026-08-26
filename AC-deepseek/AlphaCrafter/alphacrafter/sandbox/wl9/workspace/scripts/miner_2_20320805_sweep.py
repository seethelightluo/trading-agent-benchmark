"""miner_2 cycle 2032-08-05: sweep new candidate factor families.
Visible history through 2032-08-04 (last completed trading day). No lookahead.
Admission gates: abs daily paper IC >= 0.0070 and abs ICIR >= 0.084.
"""
import numpy as np, pandas as pd
from pathlib import Path
VISIBLE_END='2032-08-04'
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
closes={}; highs={}; lows={}; vols={}
for a in ASSETS:
    p=SD/f'{a}.csv'
    if not p.exists(): p=ID/f'{a}.csv'
    df=pd.read_csv(p,parse_dates=['date']); df=df[df['date']<=VISIBLE_END].sort_values('date').set_index('date')
    closes[a]=df['close'].astype(float); highs[a]=df['high'].astype(float); lows[a]=df['low'].astype(float)
    vols[a]=df['volume'].astype(float) if 'volume' in df else pd.Series(np.nan,index=df.index)
close=pd.DataFrame(closes).dropna(); high=pd.DataFrame(highs).reindex(close.index); low=pd.DataFrame(lows).reindex(close.index); vol=pd.DataFrame(vols).reindex(close.index)
rets=close.pct_change().dropna(); ret_idx=rets.index
fwd10=rets.shift(-10).rolling(10).mean(); fwd5=rets.shift(-5).rolling(5).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, {close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}")

vix=pd.read_csv(ID/'VIX.csv',parse_dates=['date']); vix=vix[vix['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
dxy=pd.read_csv(ID/'DXY.csv',parse_dates=['date']); dxy=dxy[dxy['date']<=VISIBLE_END].set_index('date')['close'].astype(float)
usdcny=pd.read_csv(ID/'USDCNY.csv',parse_dates=['date']); usdcny=usdcny[usdcny['date']<=VISIBLE_END].set_index('date')['close'].astype(float)

def compute_ic(fv, fwd, min_dates=30):
    fv=fv.reindex(ret_idx); ics=[]; n_ok=0
    for d in ret_idx:
        f=fv.loc[d]; r=fwd.loc[d]; m=f.notna()&r.notna()
        if m.sum()>=8:
            n_ok+=1; fv_=f[m].rank().values; rv_=r[m].rank().values
            if fv_.std()>0 and rv_.std()>0: ics.append(np.corrcoef(fv_,rv_)[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0,'cov':0.0}
    hit=float((ics>0).mean()); cov=float(fv.notna().mean().mean())
    mu=ics.mean(); sd=ics.std(); icir=mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov}

def turnover(fv):
    fv=fv.reindex(ret_idx); s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def report(name, fv, fwd=None):
    fwd=fwd or fwd10; ic=compute_ic(fv,fwd)
    print(f"{name}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(fv):.3f}", flush=True)

# A: Sharpe-like risk-adjusted momentum (mom20 / vol20)
mom20=pd.DataFrame({a:close[a].pct_change(20) for a in ASSETS})
vol20=pd.DataFrame({a:rets[a].rolling(20).std() for a in ASSETS})
sharpe20=mom20/vol20
report("A sharpe20(mom20/vol20)", sharpe20)

# B: return acceleration: mom20 - mom60 (second derivative)
mom60=pd.DataFrame({a:close[a].pct_change(60) for a in ASSETS})
accel=mom20-mom60
report("B accel(mom20-mom60)", accel)

# C: cross-sectional relative beta to equal-weight EW 15-asset
ew=rets.mean(axis=1)
relbeta=pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    j=pd.concat([rets[a].rename('a'), ew.rename('m')],axis=1).dropna()
    relbeta[a]=j['a'].rolling(60).cov(j['m'])/j['m'].rolling(60).var()
report("C rel_beta_ew_60", relbeta)

# D: max drawdown over 60d (downside risk proxy), use neg
dd60=pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    c=close[a]; roll=c.rolling(60).max(); dd60[a]=c/roll-1
report("D maxdd_60 (neg)", -dd60)

# E: 20d downside semi-deviation (neg returns only) -> neg
dsdev=pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
neg_rets=rets.clip(upper=0)
dsdev=neg_rets.rolling(20).std()
report("E downside_semi_std_20 (neg)", -dsdev)

# F: 20d upside/downside capture ratio (skewness-like) 
ret_pos=rets.clip(lower=0); rp=ret_pos.rolling(20).mean(); rn=neg_rets.abs().rolling(20).mean()
ud=rp/(rn+1e-9)
report("F upside_down_ratio_20", ud)

# G: volume momentum ratio 20d/60d (liquidity acceleration)
vmom=vol.rolling(20).mean()/vol.rolling(60).mean()
report("G vol_mom_20v60", vmom)

# H: 60d beta to DXY (dollar sensitivity), combined cross-asset
dxy_ret=dxy.pct_change().reindex(ret_idx).fillna(0)
dbeta=pd.DataFrame(index=ret_idx, columns=ASSETS, dtype=float)
for a in ASSETS:
    j=pd.concat([rets[a].rename('a'), dxy_ret.rename('d')],axis=1).dropna()
    dbeta[a]=j['a'].rolling(60).cov(j['d'])/j['d'].rolling(60).var()
report("H beta_dxy_60", dbeta)
