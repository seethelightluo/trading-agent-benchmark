import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    d=get_stock_daily_data(s,6000)
    if d is None or len(d)<100: d=get_index_daily_data(s,6000)
    return d.set_index(pd.to_datetime(d.date)).close.rename(s) if d is not None else pd.Series(dtype=float,name=s)
px=pd.concat([load(s) for s in U],axis=1).sort_index().ffill()
# downside-risk-adjusted medium momentum, lagged one day
ret=px.pct_change()
down=ret.where(ret<0,0).rolling(30,min_periods=20).std()*np.sqrt(252)
sig=px.pct_change(30).div(down).shift(1)
# cross-sectional demean to remove common scale
sig=sig.sub(sig.mean(axis=1),axis=0)
print('range',px.index.min(),px.index.max(),'assets',px.shape[1])
for h in [5,10,20,40]:
    fr=px.shift(-h).div(px)-1
    rows=[]; ns=[]
    for dt in sig.index:
        x=sig.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
    a=pd.Series(rows).dropna()
    print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
# turnover based ranks/signals
r=sig.rank(axis=1,pct=True); print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',r.diff().abs().mean().mean())
# regime 3 blocks at 2020-24,25-29,30-34 for 40d
h=40; fr=px.shift(-h).div(px)-1; vals=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
v=pd.Series(dict(vals));
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 q=v.loc[a:b].dropna(); print('regime',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
out=pd.DataFrame(sig); out.index.name='date'; out.to_csv('scripts/miner_1_20350105_downside_momentum_signal.csv')
