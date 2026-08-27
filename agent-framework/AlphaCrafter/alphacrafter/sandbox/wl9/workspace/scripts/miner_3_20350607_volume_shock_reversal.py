import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
cl={}; vo={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date'); cl[s]=d.close.astype(float); vo[s]=d.volume.astype(float) if 'volume' in d else pd.Series(index=d.index,dtype=float)
prices=pd.DataFrame(cl).sort_index().loc[:'2035-06-06']; volume=pd.DataFrame(vo).reindex(prices.index); r=prices.pct_change()
# Contrarian response to a short return shock, amplified by abnormal volume, risk scaled.
shock=-(prices/prices.shift(5)-1); vol=r.rolling(20,min_periods=15).std(); vs=volume/(volume.rolling(60,min_periods=30).median()+1e-12)-1
f=(shock/(vol+0.005))*(1+vs.clip(-0.8,3.0)); f=f.replace([np.inf,-np.inf],np.nan).shift(1).clip(-10,10)
def calc(h):
 vals=[]; dates=[]; ns=[]
 for i in range(len(prices)-h):
  z=pd.concat([f.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(prices.index[i]); ns.append(len(z))
 x=pd.Series(vals,index=pd.DatetimeIndex(dates)); return x,len(x),np.mean(ns)
print('universe',len(U),'dates',prices.index.min().date(),prices.index.max().date())
for h in [5,10,20,40,60]:
 x,n,an=calc(h); print('H',h,'dates',n,'avgN',round(an,2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,_,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20350607_volume_shock_reversal_signal.csv',index=False)
