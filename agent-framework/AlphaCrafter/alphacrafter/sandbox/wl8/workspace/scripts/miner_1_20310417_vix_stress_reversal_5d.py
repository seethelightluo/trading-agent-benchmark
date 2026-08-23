import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p);x.date=pd.to_datetime(x.date);D[s]=x.set_index('date').close
P=pd.concat(D,axis=1).sort_index();R=P.pct_change();
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date')['close'].reindex(P.index).ffill()
# Stress-conditioned 5d reversal: activate only when VIX is above its trailing 60d median, normalized by asset vol.
stress=(v>v.rolling(60).median()).astype(float); sig=-(P/P.shift(5)-1)/(R.rolling(20).std()*np.sqrt(5))*stress.values[:,None]
ics=[];ns=[];turn=[]
for i in range(65,len(P)-10):
 z=pd.concat([sig.iloc[i],P.iloc[i+10]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  if i>65:
   q=pd.concat([sig.iloc[i],sig.iloc[i-1]],axis=1).dropna();turn.append(np.mean(np.sign(q.iloc[:,0])!=np.sign(q.iloc[:,1])))
a=np.array(ics);print('dates',len(a),'avg_inst',np.mean(ns),'coverage',np.mean(ns)/15,'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',np.mean(a>0),'turnover',np.mean(turn))
for n in [180,360]:
 b=a[-n:];print('recent',n,len(b),b.mean(),b.mean()/b.std())
for h in [1,5,10]:
 q=[]
 for i in range(65,len(P)-h):
  z=pd.concat([sig.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('decay',h,len(q),q.mean(),q.mean()/q.std())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20310417_vix_stress_reversal_5d_signal.csv',index=False)
