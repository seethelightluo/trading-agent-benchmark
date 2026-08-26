import numpy as np,pandas as pd,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f): D[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill();r=p.pct_change(); ret20=p.pct_change(20);v20=r.rolling(20,min_periods=15).std()
med=v20.median(axis=1);vr=v20.div(med,axis=0).replace(0,np.nan).clip(.25,4)
sig=(ret20/(v20.replace(0,np.nan)*np.sqrt(252))/vr).shift(1)
fwd={h:p.shift(-h)/p-1 for h in [1,5,10,20]}
def calc(x):
 a=[];ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 return pd.Series(a),np.mean(ns)
v,n=calc(fwd[10]);print('dates',len(v),'avgN',round(n,3),'coverage',round(sig.notna().mean().mean(),5),'IC',round(v.mean(),6),'ICIR',round(v.mean()/v.std(ddof=1),6),'hit',round((v>0).mean(),5))
for h in [1,5,10,20]:
 q,_=calc(fwd[h]);print('decay',h,round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
for w in [365,750,1260]:
 q=v.tail(w);print('recent',w,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
r=sig.rank(axis=1,pct=True);ts=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:ts.append((z.iloc[:,1]-z.iloc[:,0]).abs().mean())
print('turnover',round(np.mean(ts),6),'signal_dates',sig.notna().any(axis=1).sum())
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20350104_volatility_gated_momentum_signal.csv',index=False)
