import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-01-12')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();
def macro(name):
 x=pd.read_csv('../persistent/index_data/'+name+'.csv'); x.date=pd.to_datetime(x.date); return x[x.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
dxy=macro('DXY'); vix=macro('VIX')
# all inputs shifted: signal at t uses through t-1, forward return t->t+1
r=px.pct_change(); dr=dxy.pct_change(); vr=vix.pct_change()
cond_dxy=(dr.rolling(10).sum().shift(1)<0)
cond_vix=(vr.rolling(5).sum().shift(1)>0)
base=r.rolling(20).sum().shift(1)
# test interpretable macro regimes and defensive tilt variants
sigs={'dxy_down_mom':base.where(cond_dxy,-base), 'vix_up_mom':base.where(cond_vix,-base), 'dxy_down_reversal':-r.rolling(3).sum().shift(1).where(cond_dxy,r.rolling(3).sum().shift(1)), 'vix_up_reversal':-r.rolling(3).sum().shift(1).where(cond_vix,r.rolling(3).sum().shift(1))}
fwd=px.shift(-1)/px-1
for nm,sig in sigs.items():
 vals=[]; ns=[]; dates=[]
 for dt in px.index:
  g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q): vals.append(q);ns.append(len(g));dates.append(dt)
 a=np.array(vals); print(nm,'dates',len(a),'avgN',round(np.mean(ns),2),'cov',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for label,mask in [('2020-22',pd.Series(px.index.year<=2022,index=px.index)),('2023-25',pd.Series((px.index.year>=2023)&(px.index.year<=2025),index=px.index)),('2026',pd.Series(px.index.year==2026,index=px.index)),('2027+',pd.Series(px.index.year>=2027,index=px.index)),('recent180',pd.Series(px.index>=END-pd.Timedelta(days=180),index=px.index))]:
  z=a[[m for m,d in enumerate(dates) if mask.loc[d]]]
  print(label,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
