import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data'
p=pd.DataFrame({s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date').close.astype(float) for s in U}).sort_index().loc[:'2035-09-26']; r=p.pct_change(); ret=p.shift(1)/p.shift(121)-1; v=r.rolling(120,min_periods=80).std().shift(1); path=r.abs().rolling(120,min_periods=80).sum().shift(1); e=(ret.abs()/(path+1e-12)).clip(0,1); f=(-ret/(v*np.sqrt(120)+.05))*e; f=f.clip(-20,20)
def c(h):
 a=[];d=[];n=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);d.append(p.index[i]);n.append(len(z))
 return pd.Series(a,index=d),n
for h in [5,10,20,40,60]:
 x,n=c(h);print('H',h,'dates',len(x),'avgN',round(np.mean(n),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,n=c(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]; print('REG',a,b,len(q),round(q.mean(),6))
print('coverage',round(f.notna().sum(axis=1).div(15).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20350927_long_reversal_efficiency_signal.csv',index=False)
