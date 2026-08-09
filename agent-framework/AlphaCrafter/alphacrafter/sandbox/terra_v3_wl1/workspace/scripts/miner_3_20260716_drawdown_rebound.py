import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change()
# Drawdown-rebound: distance below trailing 20-day high, with no forward information.
f=p/p.rolling(20).max()-1
f=-f
for h in [1,3,5,10,20]:
  ic=[]; ns=[]
  for i in range(20,len(p)-h):
    z=pd.concat([f.iloc[i],p.shift(-h).iloc[i]/p.iloc[i]-1],axis=1).dropna()
    if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  a=np.asarray(ic); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'ic',round(a.mean(),5),'icir',round(a.mean()/a.std(ddof=1),5),'hit',round(np.mean(a>0),4),'recent500',round(a[-500:].mean(),5),round(a[-500:].mean()/a[-500:].std(ddof=1),5))
  if h==5:
   for j,b in enumerate(np.array_split(a,5)): print('block',j+1,round(b.mean(),5),round(b.mean()/b.std(ddof=1),5))
# signal turnover as rank changes, and cross-sectional correlation to existing 5d reversal
x=f.rank(axis=1,pct=True); rev=-p.pct_change(5)
cs=[];to=[]
for i in range(21,len(p)):
 z=pd.concat([f.iloc[i],rev.iloc[i]],axis=1).dropna()
 if len(z)>=8: cs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 to.append(np.mean(abs(x.iloc[i]-x.iloc[i-1]).dropna()))
print('library_proxy_corr_to_rev5',np.nanmean(cs),'rank_turnover',np.nanmean(to))
