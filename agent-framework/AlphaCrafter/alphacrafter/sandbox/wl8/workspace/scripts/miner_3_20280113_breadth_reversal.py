import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-01-12')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change();
# Cross-sectional breadth state, lagged, applied to lagged 5d reversal; fallback sign is explicit.
cs=r.shift(1).rank(axis=1,pct=True); breadth=(cs>0.5).mean(axis=1).shift(1)
rev=-r.rolling(5).sum().shift(1)
sigs={'breadth_low_reversal':rev.where(breadth<0.35,-rev),'breadth_high_reversal':rev.where(breadth>0.65,-rev)}
fwd=px.shift(-1)/px-1
for nm,sig in sigs.items():
 vals=[]; ns=[]; dates=[]
 for dt in px.index:
  g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q):vals.append(q);ns.append(len(g));dates.append(dt)
 a=np.array(vals); print(nm,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for label,mask in [('2020-22',pd.Series(px.index.year<=2022,index=px.index)),('2023-25',pd.Series((px.index.year>=2023)&(px.index.year<=2025),index=px.index)),('2026',pd.Series(px.index.year==2026,index=px.index)),('2027+',pd.Series(px.index.year>=2027,index=px.index)),('recent180',pd.Series(px.index>=END-pd.Timedelta(days=180),index=px.index))]:
  z=a[[i for i,d in enumerate(dates) if mask.loc[d]]]; print(label,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
