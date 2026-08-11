import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').close.sort_index()
px=pd.concat(P,axis=1).sort_index().ffill(); r=px.pct_change()
# Relative momentum: asset's 20d log trend versus contemporaneous cross-sectional median,
# normalized by its own realized volatility, lagged one completed day.
raw=np.log(px/px.shift(20))
csmed=raw.median(axis=1)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
f=raw.sub(csmed,axis=0).div(vol).replace([np.inf,-np.inf],np.nan).shift(1)
print('universe',len(U),'loaded',len(P),'range',px.index.min(),px.index.max())
for h in [1,3,5,10,20]:
 out=[]; ns=[]
 fw=px.pct_change(h).shift(-h)
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):out.append(q);ns.append(len(a))
 x=pd.Series(out);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
# diagnostics at admission horizon
fw=px.pct_change(10).shift(-10); vals=[]
for dt in f.index:
 a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(a)>=8:
  q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
  if pd.notna(q):vals.append((dt,q))
x=pd.Series(dict(vals));print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
for lab,aa in [('2020-22',x.loc['2020':'2022']),('2023-25',x.loc['2023':'2025']),('2026-27',x.loc['2026':'2027']),('2028YTD',x.loc['2028':])]:print(lab,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else np.nan)
print('latest',x.index.max())
