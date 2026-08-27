import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
 px[s]=d
prices=pd.DataFrame(px).sort_index(); prices=prices.loc[:'2035-04-25']; rets=prices.pct_change()
# short shock reversal, risk scaled, computed from information through date t
shock=-(prices/prices.shift(5)-1)
vol=rets.rolling(20,min_periods=15).std()*np.sqrt(252)
f=shock/(vol+0.005)
# winsorize cross-section each day to limit crypto outliers
f=f.clip(-5,5)
def calc(h):
 vals=[]; dates=[]; ns=[]
 for i in range(len(prices)-h):
  dt=prices.index[i]; a=f.iloc[i]; y=prices.iloc[i+h]/prices.iloc[i]-1
  z=pd.concat([a,y],axis=1).dropna();
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(dates)); return x,len(ns),np.mean(ns)
print('universe',len(U),'dates',prices.index.min().date(),prices.index.max().date())
for h in [5,10,20,40,60]:
 x,n,an=calc(h); print('H',h,'dates',n,'avgN',round(an,2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
# regimes H10
x,_,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
# coverage and turnover proxy
valid=f.notna().sum(axis=1)/len(U); print('coverage',round(valid.mean(),4),'turnover_proxy',round((f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1)).mean(),4))
# artifact
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20350426_shortshock_volscaled_reversal_signal.csv',index=False)
