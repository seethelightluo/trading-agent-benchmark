import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-11-19')
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); P[s]=d.close
p=pd.DataFrame(P).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
for w in [10,20,40]:
 vals=[]; ns=[]; ranks=[]
 for s in U:
  # rolling beta to contemporaneous equal-weight universe, then residual cumulative return
  beta=r[s].rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var()
  f=r[s].rolling(w,min_periods=w).sum()-beta*m.rolling(w,min_periods=w).sum()
  y=r[s].shift(-1)
  for dt in p.index:
   if pd.notna(f.get(dt)) and pd.notna(y.get(dt)): pass
  tmp=pd.DataFrame({'f':f,'y':y}).dropna()
  for dt,g in tmp.groupby(tmp.index): pass
  # collect cross-section later
  for dt in tmp.index: pass
  if s==U[0]: pass
 # matrix aligned
 F={}; Y=r.shift(-1)
 for s in U:
  beta=r[s].rolling(60,min_periods=45).cov(m)/m.rolling(60,min_periods=45).var()
  F[s]=r[s].rolling(w,min_periods=w).sum()-beta*m.rolling(w,min_periods=w).sum()
 F=pd.DataFrame(F); z=[]; n=[]; rk=[]
 for dt in p.index:
  a=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:
   z.append(spearmanr(a.f,a.y).statistic); n.append(len(a)); rk.append(a.f.rank(pct=True))
 z=np.array(z); print('w',w,'dates',len(z),'avgN',np.mean(n),'coverage',F.notna().mean().mean(),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turn',pd.DataFrame(rk).diff().abs().mean().mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
  q=[]
  for dt,g in pd.DataFrame({'f':F.stack(),'y':Y.stack()}).groupby(level=0):
   if lo<=str(dt.year)<=hi:
    a=g.dropna()
    if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:q.append(spearmanr(a.f,a.y).statistic)
  q=np.array(q); print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
 for h in [5,10]:
  yy=r.shift(-1).rolling(h).sum().shift(-(h-1)); q=[]
  for dt in p.index:
   a=pd.DataFrame({'f':F.loc[dt],'y':yy.loc[dt]}).dropna()
   if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:q.append(spearmanr(a.f,a.y).statistic)
  q=np.array(q);print('h',h,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
