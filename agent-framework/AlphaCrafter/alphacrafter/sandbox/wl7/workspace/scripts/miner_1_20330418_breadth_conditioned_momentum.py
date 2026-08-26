import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-04-17'); D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change(fill_method=None)
ret20=px.pct_change(20,fill_method=None); vol40=r.rolling(40,min_periods=25).std(); mom=ret20/(vol40*np.sqrt(20)+1e-12)
breadth=(ret20>0).mean(axis=1); amp=(0.5+(breadth-0.5).abs()); sig=mom.mul(amp,axis=0).shift(1)
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(z): vals.append(z);ns.append(len(q))
 a=np.array(vals); print('H',h,'dates',len(a),'assets_avg',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4))
f=px.shift(-10)/px-1;o=[]
for dt in px.index:
 q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8:
  z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
  if np.isfinite(z):o.append(z)
a=np.array(o); print('thirds',[round(np.mean(x),6) for x in np.array_split(a,3) if len(x)],'coverage',round(sig.notna().mean().mean(),4),'turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4),'n_assets',len(D),'n_dates',len(px))
sig.to_csv('scripts/miner_1_20330418_breadth_conditioned_momentum_signal.csv')
