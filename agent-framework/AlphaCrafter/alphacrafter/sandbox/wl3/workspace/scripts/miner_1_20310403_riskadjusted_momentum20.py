import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.DataFrame(D).sort_index(); r=p.pct_change()
# Candidate: 20-session momentum normalized by trailing 20-session realized volatility
f=r.rolling(20).sum()/r.rolling(20).std()
rows=[]
for h in [1,3,5,10]:
  ics=[]; dates=[]; ns=[]
  for i in range(60,len(p)-h):
    x=f.iloc[i-1]; y=p.pct_change(h).iloc[i+h-1] # return from close i-1 to i+h-1? factor at i-1, forward starts i-1
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
      ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(p.index[i]); ns.append(len(z))
  a=np.array(ics); print('H',h,'dates',len(a),'avgN',np.mean(ns),'coverage',np.mean(ns)/15,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
# regime daily
h=1; ics=[]
for i in range(60,len(p)-1):
 z=pd.concat([f.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8: ics.append((p.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(ics,columns=['date','ic']).set_index('date')
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2031')]:
 q=a.loc[lo:hi].ic; print('REG',lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1))
# save full signal artifact
out=f.copy(); out.to_csv('scripts/miner_1_20310403_riskadjusted_momentum20_signal.csv',index_label='date')
