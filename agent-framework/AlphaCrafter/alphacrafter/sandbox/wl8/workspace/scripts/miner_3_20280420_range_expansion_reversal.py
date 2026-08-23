import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-04-19')
H={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); H[s]=x[x.date<=END].set_index('date').sort_index()
idx=sorted(set.intersection(*[set(v.index) for v in H.values()])); hi=pd.DataFrame({s:H[s].reindex(idx).high for s in U},index=idx); lo=pd.DataFrame({s:H[s].reindex(idx).low for s in U},index=idx); cl=pd.DataFrame({s:H[s].reindex(idx).close for s in U},index=idx)
# Negative range-expansion reversal: yesterday's high-low range relative to its lagged 20d median, contrarian across assets.
rng=(hi-lo)/cl; expansion=(rng/rng.rolling(20,min_periods=10).median()).shift(1); sig=-expansion
fwd=cl.shift(-1)/cl-1; vals=[]; dates=[]; ns=[]
for d in idx:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1 and g.f.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q): vals.append(q); dates.append(d); ns.append(len(g))
a=np.array(vals); print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sig.notna().sum().sum()/sig.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
for lab,fn in {'2020-22':lambda d:d.year<=2022,'2023-25':lambda d:2023<=d.year<=2025,'2026':lambda d:d.year==2026,'2027':lambda d:d.year==2027,'2028':lambda d:d.year>=2028,'recent180':lambda d:d>=END-pd.Timedelta(days=180)}.items():
 z=a[[i for i,d in enumerate(dates) if fn(d)]]; print(lab,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None,'dates',len(z))
for h in [2,5,10]:
 ff=cl.shift(-h)/cl-1; vv=[]
 for d in idx:
  g=pd.DataFrame({'s':sig.loc[d],'f':ff.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vv.append(spearmanr(g.s,g.f).statistic)
 z=np.array(vv); print('horizon',h,'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6),'dates',len(z))
