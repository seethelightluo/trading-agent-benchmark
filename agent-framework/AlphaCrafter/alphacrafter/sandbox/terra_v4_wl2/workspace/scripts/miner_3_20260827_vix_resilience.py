import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
px={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date').close
 px[s]=x
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close.reindex(r.index).ffill().pct_change()
# resilience: mean asset return on recent VIX-up days minus mean return on all days, rolling 60; signal known at t predicts t+1
for w in [40,60,90]:
 rows=[]
 for t in range(w,len(r)-1):
  z=r.iloc[t-w:t]; shock=vx.iloc[t-w:t]>0
  if shock.sum()<5: continue
  f=z.where(pd.DataFrame(np.tile(shock.values[:,None],(1,z.shape[1])),index=z.index,columns=z.columns), np.nan).mean()-z.mean()
  y=r.iloc[t+1]
  for s in U:
   if pd.notna(f.get(s)) and pd.notna(y.get(s)): rows.append((r.index[t],s,f[s],y[s]))
 a=pd.DataFrame(rows,columns=['date','s','f','y']); vals=[]
 for d,g in a.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append(spearmanr(g.f,g.y).statistic)
 q=np.array(vals); print('W',w,'dates',len(q),'avgN',a.groupby('date').size().mean(),'coverage',a.s.nunique()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  ds=pd.to_datetime(a.date.unique()); # recompute indexed below
 # turnover
 ff=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True); print('turnover',ff.diff().abs().mean().mean())
 for h in [5,10]:
  rr=[]
  for t in range(w,len(r)-h):
   z=r.iloc[t-w:t]; shock=vx.iloc[t-w:t]>0
   if shock.sum()<5: continue
   f=z.where(pd.DataFrame(np.tile(shock.values[:,None],(1,z.shape[1])),index=z.index,columns=z.columns),np.nan).mean()-z.mean(); y=r.iloc[t+1:t+1+h].sum()
   zz=pd.concat([f,y],axis=1).dropna()
   if len(zz)>=8: rr.append(spearmanr(zz.iloc[:,0],zz.iloc[:,1]).statistic)
  x=np.array(rr); print(' H',h,'N',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1))
 # regime daily using vals dates
 ser=pd.Series(vals,index=sorted(a.date.unique())[:len(vals)])
 print('regimes',[(yr,round(ser[ser.index.year==yr].mean(),4),int((ser.index.year==yr).sum())) for yr in range(2020,2027)])
