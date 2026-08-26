import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2033-03-20'); start=pd.Timestamp('2026-07-16');D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end];r=px.pct_change()
base=-(px/px.shift(5)-1)/(r.rolling(20).std()*np.sqrt(10)+1e-12);base=base.shift(1)
# Cross-asset dispersion regime is lagged; activate reversal only when dispersion is above its historical median.
disp=r.rolling(20).std().mean(axis=1).shift(1); gate=(disp>disp.expanding(120).median()).astype(float)
sig=base.mul(gate,axis=0)
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[];ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(v):vals.append(v);ns.append(len(q))
 a=np.array(vals);print('H',h,'dates',len(a),'assets_avg',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
f=px.shift(-10)/px-1;o=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
  if np.isfinite(v):o.append(v)
a=np.array(o);print('thirds',[round(x,6) for x in np.array_split(a,3) if len(x) for x in [np.mean(x)]],'coverage',round(sig.notna().mean().mean(),4),'turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4),'n_assets',len(D),'n_dates',len(px))
sig.to_csv('scripts/miner_3_20330321_dispersion_reversal_signal.csv')
