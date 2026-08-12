import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
a=[]
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<120:d=get_index_daily_data(s,1800)
 if d is not None:
  x=d[['date','close']].copy();x['symbol']=s;a.append(x)
p=pd.concat(a).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=np.log(p).diff()
vol=r.rolling(20).std(); breadth=(p.pct_change(20)>0).mean(axis=1)
# Defensive low-vol quality: low vol, with stronger preference in weak breadth, lagged
f=(-vol*(1+2*(1-breadth).values[:,None])).shift(1)
ics=[]; tr=[]; ns=[]; prev=None
for i in range(len(p)-10):
 ok=f.iloc[i].notna() & p.iloc[i+10].notna()
 if ok.sum()>=8:
  z=f.iloc[i][ok]; y=(p.iloc[i+10][ok]/p.iloc[i][ok]-1);ics.append(z.corr(y));ns.append(ok.sum());q=z.rank(pct=True);tr.append(np.nan if prev is None else (q-prev.reindex(q.index)).abs().mean());prev=q
x=np.array([v for v in ics if np.isfinite(v)]);print('candidate=stress_conditioned_lowvol; dates',len(x),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15);print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.nanmean(tr)))
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1;v=[]
 for i in range(len(p)-h):
  ok=f.iloc[i].notna()&yy.iloc[i].notna()
  if ok.sum()>=8:v.append(f.iloc[i][ok].corr(yy.iloc[i][ok]))
 v=np.array([q for q in v if np.isfinite(q)]);print('decay',h,'%.6f'%v.mean(),'n',len(v))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_3_20320304_stress_lowvol_signal.csv',index=False)
