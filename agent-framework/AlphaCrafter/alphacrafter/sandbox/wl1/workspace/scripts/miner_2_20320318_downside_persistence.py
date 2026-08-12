import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<120:d=get_index_daily_data(s,days=1800)
 if d is not None:
  z=d[['date','close']].copy();z['symbol']=s;rows.append(z)
p=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=np.log(p).diff()
# downside-adjusted persistence: medium return rewarded when downside risk is low, all lagged
ret30=p.pct_change(30); down=(r.clip(upper=0)**2).rolling(30).mean()**.5; total=r.rolling(30).std()
f=(ret30/(down+1e-8))*(1/(1+total*10)); f=f.shift(1)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; vals=[];ns=[];turn=[];prev=None
 for i in range(len(p)-h):
  ok=f.iloc[i].notna()&y.iloc[i].notna()
  if ok.sum()>=8:
   vals.append(f.iloc[i][ok].corr(y.iloc[i][ok]));ns.append(ok.sum());q=f.iloc[i].rank(pct=True);turn.append(np.nan if prev is None else (q-prev).abs().mean());prev=q
 a=np.array([x for x in vals if np.isfinite(x)])
 print('h',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(turn),4))
 if h==10:
  for lo,hi in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-03-18')]:
   vv=[]
   for dt in p.loc[lo:hi].index:
    ok=f.loc[dt].notna()&y.loc[dt].notna()
    if ok.sum()>=8:vv.append(f.loc[dt][ok].corr(y.loc[dt][ok]))
   vv=np.array([v for v in vv if np.isfinite(v)]);print('regime',lo[:4],len(vv),round(vv.mean(),6),round(vv.mean()/vv.std(ddof=1),6))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20320318_downside_persistence_signal.csv',index=False)
json_meta={'factor': 'downside_adjusted_persistence'}
print('artifact rows',len(out))
