import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').set_index('date').sort_index() for s in U}
for vb in [10,40,60]:
 rows=[]
 for s,x in D.items():
  r=x.close.pct_change(); v=r.rolling(vb,min_periods=max(5,vb//2)).std(); f=-r.rolling(3,min_periods=3).sum()/v
  rows.append(pd.DataFrame({'date':x.index,'f':f,'y':r.shift(-1),'s':s}).reset_index(drop=True))
 a=pd.concat(rows,ignore_index=True).dropna(); obs=[]
 for dt,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1:
   z=spearmanr(g.f,g.y).statistic
   if pd.notna(z):obs.append((dt,z,len(g)))
 o=pd.DataFrame(obs,columns=['date','ic','n']); x=o.ic
 ranks=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True).sort_index()
 print('vb',vb,'dates',len(o),'avgN',o.n.mean(),'coverage',len(o)/a.date.nunique(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turn',ranks.diff().abs().mean(axis=1).mean())
 for h in [5,10]:
  vals=[]
  for s,xx in D.items():
   r=xx.close.pct_change();v=r.rolling(vb,min_periods=max(5,vb//2)).std();f=-r.rolling(3,min_periods=3).sum()/v;y=xx.close.pct_change(h).shift(-h)
   vals.append(pd.DataFrame({'date':xx.index,'f':f,'y':y}).reset_index(drop=True))
  b=pd.concat(vals,ignore_index=True).dropna(); zz=[]
  for dt,g in b.groupby('date'):
   if len(g)>=8 and g.f.nunique()>1:zz.append(spearmanr(g.f,g.y).statistic)
  print(' decay',h,'IC',np.nanmean(zz),'ICIR',np.nanmean(zz)/np.nanstd(zz,ddof=1),'dates',len(zz))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=o[(o.date.dt.year>=lo)&(o.date.dt.year<=hi)].ic;print(' regime',lo,hi,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
