import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-08-22')
P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();P[a]=d.close.astype(float)
p=pd.DataFrame(P).loc[:END];r=p.pct_change();v=r.rolling(20,min_periods=15).std()
r10=p.pct_change(10);r40=p.pct_change(40)
sig=(-r10.sub(r10.mean(axis=1),axis=0)+.30*r40.sub(r40.mean(axis=1),axis=0)).div(v).shift(1)
print('cutoff',p.index.max().date(),'dates',len(p),'assets',len(A),'coverage',sig.notna().mean().mean())
ics={}
for h in [1,5,10,20]:
 f=p.pct_change(h).shift(-h);z=[];ns=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 x=np.array(z);ics[h]=x;print('h',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'dates',len(x),'mean_n',np.mean(ns))
for lo,hi in [('2020','2023'),('2023','2026'),('2026','2030')]:
 f=p.pct_change(10).shift(-10);z=[]
 for d in sig.loc[lo:hi].index:
  q=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 x=np.array(z);print('regime',lo,hi,'n',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
rk=sig.rank(axis=1,pct=True);print('turnover10',rk.diff(10).abs().mean(axis=1).mean())
# Broad library correlation evidence: independent price-signal reconstructions of admitted families.
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol; rev=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(); acc=(p/p.shift(20)-p.shift(20)/p.shift(60))/vol
es=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A}); kurt=-r.rolling(40,min_periods=30).kurt()
spx=r.SPX; beta=pd.DataFrame({a:-r[a].rolling(20,min_periods=15).cov(spx)/spx.rolling(20,min_periods=15).var() for a in A})
lib={'trend':trend,'reversal':rev,'acceleration':acc,'expected_shortfall':es,'inverse_kurtosis':kurt,'negative_spx_beta':beta}
mx=0;who='';cells=0
for n,s in lib.items():
 q=pd.concat([sig.stack().rename('x'),s.stack().rename('y')],axis=1).dropna();rho=q.x.corr(q.y,method='spearman');print('library',n,'rho',rho,'cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('MAX_LIBRARY_CORR',mx,who,cells)
