import pandas as pd,numpy as np,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; start=pd.Timestamp('2026-07-16'); end=pd.Timestamp('2033-05-29'); D={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[a]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().loc[start:end]; r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
base=-(r.rolling(5,min_periods=5).sum()/(vol+1e-12)); disp=r.std(axis=1).rolling(20,min_periods=15).mean(); med=disp.rolling(120,min_periods=60).median(); scale=(disp/(med+1e-12)).clip(.25,4)
scaled=base.mul(scale,axis=0)
# cross-sectional residual of scaled shock after removing base shock, lagged one session
out=[]
for dt in px.index:
 y=scaled.loc[dt]; x=base.loc[dt]; q=pd.concat([y,x],axis=1).dropna()
 z=pd.Series(np.nan,index=px.columns)
 if len(q)>=8 and q.iloc[:,1].var()>1e-14:
  b=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/q.iloc[:,1].var()
  z.loc[q.index]=q.iloc[:,0]-b*q.iloc[:,1]
 out.append(z)
sig=pd.DataFrame(out,index=px.index).shift(1)
print('assets',len(D),'dates',len(px),'avgN',round(sig.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]:
 f=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in px.index:
  q=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if np.isfinite(c): vals.append(c);ns.append(len(q))
 a=np.array(vals); print('H',h,'valid_dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12),6),'hit',round((a>0).mean(),4),'thirds',[round(float(x.mean()),6) for x in np.array_split(a,3)])
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round((sig.rank(axis=1).diff().abs().stack()/15).mean(),4))
sig.to_csv('scripts/miner_3_20330530_residual_dispersion_shock_signal.csv')
