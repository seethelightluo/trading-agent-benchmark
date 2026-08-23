import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2027-04-02');q={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);q[s]=d[d.date<=cut].set_index('date').close
p=pd.concat(q,axis=1).sort_index();r=p.pct_change()
# residual short-term reversal, volatility damped
f=(-p.pct_change(5)/(r.rolling(20).std()*np.sqrt(20)+1e-8)).shift(1); rows=[]; ns=[]
for i in range(len(p)-1):
 v=f.iloc[i].notna()&r.iloc[i+1].notna()
 if v.sum()>=8:rows.append(spearmanr(f.iloc[i][v],r.iloc[i+1][v]).statistic);ns.append(v.sum())
x=np.array(rows);print('dates',len(x),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',np.mean(x>0))
for h in [5,10,20]:
 rr=p.pct_change(h);z=[]
 for i in range(len(p)-h):
  v=f.iloc[i].notna()&rr.iloc[i+h].notna()
  if v.sum()>=8:z.append(spearmanr(f.iloc[i][v],rr.iloc[i+h][v]).statistic)
 print('decay',h,np.mean(z))
f.index.name='date';f.to_csv('scripts/miner_1_20270402_short_reversal_signal.csv')
