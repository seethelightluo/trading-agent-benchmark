import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-02-09')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); fwd=px.shift(-1)/px-1
m=r['SPX']; beta=r.rolling(60).cov(m).div(m.rolling(60).var(),axis=0); res=r.sub(beta.mul(m,axis=0),axis=0); sig=(-res.rolling(3).sum()).shift(1)
vals=[]; dates=[]; ns=[]
for dt in px.index:
 g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q);dates.append(dt);ns.append(len(g))
a=np.array(vals); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
for lab,mk in [('2020-22',px.index.year<=2022),('2023-25',(px.index.year>=2023)&(px.index.year<=2025)),('2026',px.index.year==2026),('2027+',px.index.year>=2027),('recent180',px.index>=END-pd.Timedelta(days=180))]:
 z=a[[i for i,d in enumerate(dates) if bool(mk[px.index.get_loc(d)])]]; print(lab,round(z.mean(),6) if len(z) else None,round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,len(z))
print('turnover_proxy',round(float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
