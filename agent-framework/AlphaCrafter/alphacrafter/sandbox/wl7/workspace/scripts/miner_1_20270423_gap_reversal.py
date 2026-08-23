import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
O={};C={};H={};L={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv';
 if not os.path.exists(p): p='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); O[s]=x.open;C[s]=x.close;H[s]=x.high;L[s]=x.low
op=pd.DataFrame(O); cl=pd.DataFrame(C); hi=pd.DataFrame(H); lo=pd.DataFrame(L)
# fade prior session open-to-close gap, normalized by 20d range; strictly lagged
prev=cl.shift(1); gap=op/prev-1
rng=(hi-lo)/cl
sig=(-gap/rng.rolling(20).mean()).shift(1)
cl=cl.loc[:'2027-04-23']; sig=sig.reindex(cl.index)
for h in [1,5,10]:
 f=cl.shift(-h)/cl-1; arr=[]; ns=[]
 for d in sig.index:
  z=pd.concat([sig.loc[d],f.loc[d]],axis=1).dropna()
  if len(z)>=8: arr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(arr); print('h',h,'dates',len(a),'avg_n',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0))
 # periods
 for start,end in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-04-23')]:
  ix=[i for i,d in enumerate(sig.index) if str(d.date())>=start and str(d.date())<=end]
  q=a[[i for i in ix if i<len(a)]]
  print(start,len(q),np.mean(q) if len(q) else np.nan,np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
print('coverage',sig.notna().mean().mean(),'artifact scripts/miner_1_20270423_gap_reversal_signal.csv')
sig.reset_index().to_csv('scripts/miner_1_20270423_gap_reversal_signal.csv',index=False)
