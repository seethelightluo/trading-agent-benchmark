import pandas as pd,numpy as np,glob,json,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).astype(float); r=p.pct_change(); res=r.sub(r.mean(axis=1),axis=0)
f=(-res.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20)+1e-8)).shift(1)
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); kurt=-r.rolling(40,min_periods=30).kurt(); es=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
peer=r.sub(r.mean(axis=1),axis=0)
lib={'risk_adjusted_trend':trend,'volnorm_reversal':rev,'inverse_excess_kurtosis':kurt,'inverse_expected_shortfall':es,'ravmom':trend}
# conditional peer correlation controls
for nm,cond,sgn in [('downside_peer_correlation',peer<0,1),('inverse_upside_peer_correlation',peer>0,-1)]:
 q=pd.DataFrame(index=r.index,columns=A,dtype=float)
 for a in A:
  for i in range(39,len(r)):
   x=r[a].iloc[i-39:i+1]; y=peer[a].iloc[i-39:i+1]; m=cond[a].iloc[i-39:i+1]
   if m.sum()>=12:q.iloc[i,q.columns.get_loc(a)]=sgn*x[m].corr(y[m])
 lib[nm]=q
mx=0;who='none'; ev={}
for n,s in lib.items():
 z=pd.concat([f.stack().rename('candidate'),s.stack().rename('library')],axis=1).dropna(); rho=z.candidate.corr(z.library,method='spearman');ev[n]=float(rho)
 if abs(rho)>mx:mx=abs(rho);who=n
 print('LIB',n,'rho',round(rho,6),'cells',len(z))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_COMPLETE',True)
# forward IC
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(ss);print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'dates',len(s),'N',round(np.mean(ns),2),'hit',round((s>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6))
print('EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
