import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-02-24'); X={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index();p=p[p.index<=cutoff];r=p.pct_change();down=r.clip(upper=0).rolling(20,min_periods=15).std();f=(p.pct_change(10)/(down*np.sqrt(10))).replace([np.inf,-np.inf],np.nan)
 X[s]=pd.DataFrame({'f':f,'r1':p.shift(-1)/p-1,'r5':p.shift(-5)/p-1,'r10':p.shift(-10)/p-1})
dates=sorted(set.intersection(*[set(v.index) for v in X.values()]))
for h in ['r1','r5','r10']:
 a=[];ns=[]
 for d in dates:
  z=pd.Series({s:X[s].loc[d,'f'] for s in U});y=pd.Series({s:X[s].loc[d,h] for s in U});q=z.notna()&y.notna()
  if q.sum()>=8 and z[q].nunique()>1:a.append(spearmanr(z[q],y[q]).statistic);ns.append(q.sum())
 a=np.array(a);print(h,'dates',len(a),'avgN',np.mean(ns),'IC %.8f ICIR %.8f hit %.4f coverage %.4f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),np.mean(ns)/15))
print('period',dates[0],dates[-1])
