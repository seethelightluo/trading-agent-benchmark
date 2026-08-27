import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'; cl={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date'); cl[s]=d.close.astype(float)
prices=pd.DataFrame(cl).sort_index().loc[:'2035-08-15']; ret=prices.pct_change()
# Contrarian 20D return, scaled by downside volatility asymmetry. All inputs lagged one
# completed day: downside/upside 60D semideviation ratio, capped for stability.
down=ret.where(ret<0,0).rolling(60,min_periods=40).std()
up=ret.where(ret>0,0).rolling(60,min_periods=40).std()
asym=(down/(up+1e-4)).clip(0.25,4.0)
f=(-(prices.shift(1)/prices.shift(21)-1)/(ret.rolling(60,min_periods=40).std().shift(1)*np.sqrt(60)+0.05))*asym.shift(1)
f=f.replace([np.inf,-np.inf],np.nan).clip(-20,20)
def calc(h):
 vals=[]; dates=[]; ns=[]
 for i in range(len(prices)-h):
  z=pd.concat([f.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(prices.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(dates)); return x,ns
print('universe',len(U),'dates',prices.index.min().date(),prices.index.max().date())
for h in [5,10,20,40,60]:
 x,ns=calc(h); print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'minN',min(ns),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,ns=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20350816_downside_asymmetry_reversal_signal.csv',index=False)
