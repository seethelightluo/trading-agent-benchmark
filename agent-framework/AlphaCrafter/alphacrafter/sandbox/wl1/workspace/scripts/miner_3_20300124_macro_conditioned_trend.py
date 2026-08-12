import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={a:get_stock_daily_data(a,days=5000) for a in U}
px=pd.DataFrame({a:x.set_index('date')['close'] for a,x in D.items()})
V=get_index_daily_data('VIX',days=5000).set_index('date')['close'].rename('vix')
f=pd.DataFrame(index=px.index)
for a in U:
 s=px[a].dropna(); rr=s.pct_change(); vv=rr.rolling(20,min_periods=15).std(); b=rr.rolling(60,min_periods=45).sum()/(vv*np.sqrt(60)+1e-12)
 f[a]=b.reindex(px.index)
vr=V.reindex(px.index).ffill(); vz=(vr-vr.rolling(120,min_periods=80).median())/(vr.rolling(120,min_periods=80).std()+1e-12)
f=f*(1+0.35*np.tanh(vz.clip(-3,3)).values[:,None])
rows=[]
for t in px.index:
 for a in U:
  if pd.notna(f.loc[t,a]):
   s=px[a].loc[t:].iloc[1:21]
   for h in [1,5,10,20]:
    if len(s)>=h and pd.notna(s.iloc[h-1]) and pd.notna(px.loc[t,a]): rows.append((t,a,f.loc[t,a],h,s.iloc[h-1]/px.loc[t,a]-1))
out=pd.DataFrame(rows,columns=['date','asset','factor','h','fwd'])
print('rows',len(out),'dates',out.date.nunique(),'assets',out.asset.nunique())
for h,g in out.groupby('h'):
 cs=g.groupby('date').filter(lambda z:len(z)>=8).groupby('date').apply(lambda z:z.factor.corr(z.fwd)).dropna()
 print('METRIC',h,len(cs),g.asset.nunique(),round(cs.mean(),6),round(cs.mean()/cs.std(ddof=1),6),round((cs>0).mean(),4))
for label,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-28','2026-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31')]:
 z=out[(out.h==20)&(out.date>=lo)&(out.date<=hi)].groupby('date').apply(lambda q:q.factor.corr(q.fwd)).dropna(); print('REGIME',label,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6),round((z>0).mean(),4))
out[out.h==20][['date','asset','factor']].to_csv('scripts/miner_3_20300124_macro_conditioned_trend_signal.csv',index=False)
