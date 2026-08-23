import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2031-05-15')
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.set_index('date').close
P=pd.concat(D,axis=1).sort_index(); P=P.loc[P.index<=cutoff]; r=P.pct_change()
ret=P/P.shift(10)-1; path=r.abs().rolling(20).sum(); sig=-(ret/path).replace([np.inf,-np.inf],np.nan)
ics=[]; ns=[]; turns=[]
for i in range(80,len(P)-10):
 z=pd.concat([sig.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  if i>80:
   q=pd.concat([sig.iloc[i],sig.iloc[i-1]],axis=1).dropna(); turns.append(np.mean(np.sign(q.iloc[:,0])!=np.sign(q.iloc[:,1])))
a=np.array(ics); print('period',P.index[80].date(),P.index[-1].date(),'dates',len(a),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0),'turnover',np.mean(turns))
for n in [180,360]:
 b=a[-n:]; print('recent',n,'IC',b.mean(),'ICIR',b.mean()/b.std(),'hit',np.mean(b>0))
for h in [1,5,10]:
 q=[]
 for i in range(80,len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[i*0 if False else 1]).statistic)
 q=np.array(q); print('decay',h,len(q),q.mean(),q.mean()/q.std())
# latest signal artifact
sig.tail(1).T.rename(columns={sig.index[-1]:'signal'}).to_csv('scripts/miner_1_20310515_efficiency_reversal_signal.csv')
