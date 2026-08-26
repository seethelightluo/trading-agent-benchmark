import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv(os.path.join('../persistent/stock_data',s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index()
P=P.loc[:'2033-08-03']; r=np.log(P/P.shift(1)); ret20=P/P.shift(20)-1
cons=r.rolling(20).mean().abs()/r.rolling(20).std(); factor=ret20*cons
ics={}; ns=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],(P.shift(-10)/P-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): ics[dt]=q; ns.append(len(z))
a=pd.Series(ics).sort_index(); sub=a.iloc[::10]
print('dates',len(a),'nonoverlap',len(sub),'avgN',np.mean(ns),'coverage',factor.notna().sum(axis=1).mean()/15,'last',P.index.max())
for label,q in [('all',a),('nonoverlap',sub),('recent365',a[a.index>=a.index.max()-pd.Timedelta(days=365)]),('recent750',a[a.index>=a.index.max()-pd.Timedelta(days=750)])]: print(label,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q))
for h in [1,5,10,20]:
 f=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],(P.shift(-h)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8:f.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(f),len(f))
print('turnover_proxy',factor.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
