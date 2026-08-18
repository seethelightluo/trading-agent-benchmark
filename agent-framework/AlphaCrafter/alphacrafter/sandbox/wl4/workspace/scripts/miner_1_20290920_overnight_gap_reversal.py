import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cutoff=pd.Timestamp('2029-09-19')
o={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'));d.date=pd.to_datetime(d.date);d=d[d.date<=cutoff].set_index('date').sort_index();o[s]=d
P=pd.DataFrame({s:d.close for s,d in o.items()}).sort_index(); O=pd.DataFrame({s:d.open for s,d in o.items()}).reindex(P.index)
# Gap reversal: prior close-to-open shock, lagged one session; use prior close/open and normalize by 20d vol.
gap=O/P.shift(1)-1; v=P.pct_change().rolling(20,min_periods=15).std(); sig=(-gap/(v+1e-12)).shift(1)
print('rows',len(P),'range',P.index.min().date(),P.index.max().date())
for h in [1,5,10,20]:
 f=P.shift(-h)/P-1; a=[]; ns=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):a.append(q);ns.append(len(z))
 a=np.array(a);print(h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
 for n in [250,500]:
  if len(a)>=n:
   q=a[-n:];print(' recent',n,'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(n),6))
print('turnover_proxy',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'panel_valid',round(sig.notna().sum().sum()/sig.size,4))
