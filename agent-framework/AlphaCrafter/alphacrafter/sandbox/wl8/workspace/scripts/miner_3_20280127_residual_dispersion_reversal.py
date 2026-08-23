import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-01-26')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); r=px.pct_change(); med=r.median(axis=1); resid=r.sub(med,axis=0); disp=r.std(axis=1).rolling(5).mean()
sig=(-resid.div(disp,axis=0)).shift(1).clip(-10,10); fwd=px.shift(-1)/px-1
vals=[]; ns=[]; dates=[]
for dt in px.index:
 g=pd.DataFrame({'s':sig.loc[dt],'f':fwd.loc[dt]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q);ns.append(len(g));dates.append(dt)
a=np.array(vals); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
regimes={'2020-22':lambda d:d.year<=2022,'2023-25':lambda d:2023<=d.year<=2025,'2026':lambda d:d.year==2026,'2027+':lambda d:d.year>=2027,'recent180':lambda d:d>=END-pd.Timedelta(days=180)}
for label,fn in regimes.items():
 z=a[[i for i,d in enumerate(dates) if fn(d)]]; print(label,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'n',len(z))
for h in [1,3,5]:
 ff=px.shift(-h)/px-1; vv=[]
 for dt in px.index:
  g=pd.DataFrame({'s':sig.loc[dt],'f':ff.loc[dt]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vv.append(spearmanr(g.s,g.f).statistic)
 print('horizon',h,'IC',round(np.nanmean(vv),6),'ICIR',round(np.nanmean(vv)/np.nanstd(vv,ddof=1),6),'dates',len(vv))
