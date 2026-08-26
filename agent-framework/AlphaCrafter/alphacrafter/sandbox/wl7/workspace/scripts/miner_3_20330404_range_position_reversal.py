import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-04-03');D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change()
# Reversal from position in recent range, lagged one completed session; normalize by volatility.
lo=px.rolling(40,min_periods=30).min(); hi=px.rolling(40,min_periods=30).max()
range_pos=(px-lo)/(hi-lo+1e-12)
sig=(0.5-range_pos)/(r.rolling(20,min_periods=15).std()+1e-12)
sig=sig.shift(1)
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[];ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(v): vals.append(v);ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'assets_avg',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
f=px.shift(-10)/px-1; o=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  v=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
  if np.isfinite(v):o.append(v)
a=np.array(o); print('thirds',[round(np.mean(x),6) for x in np.array_split(a,3) if len(x)],'coverage',round(sig.notna().mean().mean(),4),'turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4),'n_assets',len(D),'n_dates',len(px))
sig.to_csv('scripts/miner_3_20330404_range_position_reversal_signal.csv')
