import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=None
    try: d=get_index_daily_data(s,days=6000)
    except Exception: pass
    if d is None or len(d)<150:
        try: d=get_stock_daily_data(s,days=6000)
        except Exception: d=None
    if d is not None: px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); cs=r.mean(axis=1); res=r.sub(cs,axis=0)
# Contrarian response to asset-specific downside shocks, with volatility normalization.
vol=r.rolling(20,min_periods=10).std(); downside=res.where(res<0,0.0)
f=(-downside.rolling(5,min_periods=5).sum()).div(vol.shift(1)).shift(1)
print('universe',len(px),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [5,10,20,40]:
    fr=P.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
    v=np.array(vals,float); print(f'{h}D dates={len(v)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={np.nanmean(v):.8f} ICIR={np.nanmean(v)/np.nanstd(v,ddof=1)*np.sqrt(252):.8f} hit={np.mean(v>0):.4f}')
    for a,b in [('2020','2027'),('2027','2032'),('2032','2035-02-01')]:
        q=v[(pd.DatetimeIndex(dates)>=a)&(pd.DatetimeIndex(dates)<b)]
        print(' regime',a,b,'n',len(q),'IC',np.nanmean(q) if len(q) else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage_all',f.notna().sum().sum()/(len(f)*15),'turnover',rank.diff().abs().mean(axis=1).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20350215_downside_residual_reversal_signal.csv',index=False)
