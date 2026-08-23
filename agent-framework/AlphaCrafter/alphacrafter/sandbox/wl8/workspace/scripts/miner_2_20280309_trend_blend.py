import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-03-08')
P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); ret=px.pct_change()
# Interpretable medium-horizon trend blend: lagged 5d and 20d returns, volatility scaled, with recent trend receiving higher weight.
vol=ret.rolling(20,min_periods=15).std()*np.sqrt(20)
f=(0.6*ret.rolling(5,min_periods=5).sum()+0.4*ret.rolling(20,min_periods=15).sum())/vol
f=f.shift(1).clip(-10,10); fw=px.shift(-1)/px-1
vals=[]; ds=[]; ns=[]
for d in px.index:
 g=pd.DataFrame({'x':f.loc[d],'y':fw.loc[d]}).dropna()
 if len(g)>=8:
  q=spearmanr(g.x,g.y).statistic
  if np.isfinite(q): vals.append(q);ds.append(d);ns.append(len(g))
a=np.array(vals); print('factor=vol_scaled_trend_blend_5_20');print('dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(f.notna().sum().sum()/f.size,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()),6))
for lab,fn in [('2020-22',lambda d:d.year<=2022),('2023-25',lambda d:2023<=d.year<=2025),('2026',lambda d:d.year==2026),('2027',lambda d:d.year==2027),('2028YTD',lambda d:d.year==2028),('recent180',lambda d:d>=END-pd.Timedelta(days=180))]:
 z=a[[i for i,d in enumerate(ds) if fn(d)]];print(lab,'n',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6) if len(z)>1 else np.nan)
for h in [2,3,5,10]:
 fw_h=px.shift(-h)/px-1; z=[]
 for d in px.index:
  g=pd.DataFrame({'x':f.loc[d],'y':fw_h.loc[d]}).dropna()
  if len(g)>=8:z.append(spearmanr(g.x,g.y).statistic)
 z=np.array(z);print('h',h,'IC',round(np.nanmean(z),6),'ICIR',round(np.nanmean(z)/np.nanstd(z,ddof=1),6))
