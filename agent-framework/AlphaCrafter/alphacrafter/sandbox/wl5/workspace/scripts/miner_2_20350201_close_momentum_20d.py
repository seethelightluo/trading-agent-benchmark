import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ic(a,b): return pd.Series(a).rank().corr(pd.Series(b).rank())
rows=[]; sig=[]
for s in U:
 d=get_stock_daily_data(s,2800)
 if d is None: continue
 d=d.sort_values('date').reset_index(drop=True); ret=d.close.pct_change(); vol=ret.rolling(60,min_periods=30).std(); f=(d.close.pct_change(20)/vol).clip(-8,8); fr=d.close.shift(-10)/d.close-1
 for i in range(len(d)-10):
  if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): rows.append((d.date.iloc[i],f.iloc[i],fr.iloc[i])); sig.append((d.date.iloc[i],s,f.iloc[i]))
x=pd.DataFrame(rows,columns=['date','f','r']); a=[ic(g.f,g.r) for _,g in x.groupby('date') if len(g)>=8]; a=np.array([v for v in a if pd.notna(v)])
print('dates',len(a),'meanN',round(x.groupby('date').size().mean(),3),'coverage',round(len(x)/(len(a)*15),4),'IC10',round(a.mean(),6),'ICIR_daily',round(a.mean()/a.std(ddof=1)*np.sqrt(252),6),'hit',round((a>0).mean(),4))
for h in [5,20]:
 z=[]
 for s in U:
  d=get_stock_daily_data(s,2800); d=d.sort_values('date').reset_index(drop=True); ret=d.close.pct_change(); vol=ret.rolling(60,min_periods=30).std(); f=(d.close.pct_change(20)/vol).clip(-8,8); fr=d.close.shift(-h)/d.close-1
  for i in range(len(d)-h):
   if pd.notna(f.iloc[i]) and pd.notna(fr.iloc[i]): z.append((d.date.iloc[i],f.iloc[i],fr.iloc[i]))
 z=pd.DataFrame(z,columns=['date','f','r']); aa=[ic(g.f,g.r) for _,g in z.groupby('date') if len(g)>=8]; print('decay',h,round(float(np.nanmean(aa)),6),len(aa))
r=pd.DataFrame(sig,columns=['date','symbol','signal']).pivot(index='date',columns='symbol',values='signal').rank(axis=1,pct=True); print('turnover',round(float(r.diff().abs().mean(axis=1).mean()),6)); pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20350201_close_momentum_20d_signal.csv',index=False)
