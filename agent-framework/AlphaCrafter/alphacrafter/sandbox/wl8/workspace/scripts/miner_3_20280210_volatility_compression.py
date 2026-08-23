import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-09')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
# volatility compression: lagged short/long realized vol, interpreted as lower risk / rebound after unusually quiet assets
rv5=r.rolling(5).std(); rv20=r.rolling(20).std()
sig=-(rv5/rv20-1).shift(1)
# alternative volatility-adjusted recent move, same idea but controls for current risk
sig2=(-r.rolling(3).sum()/rv20).shift(1)
for nm,s in [('vol_compression',sig),('volnorm_reversal',sig2)]:
 vals=[]; dates=[]; ns=[]
 for dt in px.index:
  g=pd.DataFrame({'s':s.loc[dt],'f':fwd.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1:
   q=spearmanr(g.s,g.f).statistic
   if np.isfinite(q): vals.append(q);dates.append(dt);ns.append(len(g))
 a=np.array(vals); print(nm,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(s.notna().sum().sum()/s.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lab,m in [('2020-22',px.index.year<=2022),('2023-25',(px.index.year>=2023)&(px.index.year<=2025)),('2026',px.index.year==2026),('2027+',px.index.year>=2027),('recent180',px.index>=END-pd.Timedelta(days=180))]:
  z=a[[i for i,d in enumerate(dates) if m[px.index.get_loc(d)]]]
  print(lab,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
 # turnover average rank order change
 ranks=s.rank(axis=1,pct=True); turn=(ranks.diff().abs().mean(axis=1)).dropna().mean(); print('turnover_proxy',round(float(turn),6))
