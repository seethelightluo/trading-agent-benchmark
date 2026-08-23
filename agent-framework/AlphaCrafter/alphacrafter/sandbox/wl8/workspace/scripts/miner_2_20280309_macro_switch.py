import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-08')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index();r=px.pct_change()
def macro(n):
 x=pd.read_csv('../persistent/index_data/'+n+'.csv');x.date=pd.to_datetime(x.date);return x[x.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill()
v=macro('VIX'); d=macro('DXY'); vr=v.pct_change(); dr=d.pct_change()
# VIX shock regime: reversal on asset 3d return after a lagged VIX jump, momentum otherwise.
base=r.rolling(3,min_periods=3).sum().shift(1); m=(vr.rolling(5,min_periods=5).sum().shift(1)>0.04)
f=base.where(m,-base); fw=px.shift(-1)/px-1
for name,sig in [('vix_shock_switch',f),('dxy_trend_switch',base.where(dr.rolling(10).sum().shift(1)<0,-base))]:
 a=[];ds=[];ns=[]
 for dt in px.index:
  g=pd.DataFrame({'x':sig.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(g)>=8 and g.x.nunique()>1:
   q=spearmanr(g.x,g.y).statistic
   if np.isfinite(q):a.append(q);ds.append(dt);ns.append(len(g))
 a=np.array(a);print(name,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
 for lab,fn in [('2026',lambda x:x.year==2026),('2027',lambda x:x.year==2027),('2028',lambda x:x.year==2028),('recent180',lambda x:x>=END-pd.Timedelta(days=180))]:
  z=a[[i for i,x in enumerate(ds) if fn(x)]];print(lab,len(z),round(z.mean(),6) if len(z) else np.nan,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
