import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2033-08-03']
r=np.log(P/P.shift(1)); vol=r.rolling(20).std(); f=-(r.rolling(3).mean()/vol).rolling(3).mean(); fwd=P.shift(-10)/P-1
ics={}; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):ics[dt]=q;ns.append(len(z))
a=pd.Series(ics).sort_index(); sub=a.iloc[::10]
print('dates',len(a),'nonoverlap',len(sub),'avgN',np.mean(ns),'coverage',f.notna().sum(axis=1).mean()/15,'last',P.index.max())
for l,q in [('all',a),('nonoverlap',sub),('recent365',a[a.index>=a.index.max()-pd.Timedelta(days=365)]),('recent750',a[a.index>=a.index.max()-pd.Timedelta(days=750)])]:print(l,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q))
for h in [1,5,10,20]:
 z=[]; y=P.shift(-h)/P-1
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z),len(z))
print('turnover_proxy',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
