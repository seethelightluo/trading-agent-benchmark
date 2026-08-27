import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
prices=pd.DataFrame(px).sort_index().loc[:'2035-06-06']; ret=prices.pct_change()
# Volatility-shock reversal: recent 5D move, amplified when short vol exceeds its 60D baseline;
# normalize by 20D vol and lag one session. Positive means expected rebound.
v20=ret.rolling(20,min_periods=15).std(); v60=ret.rolling(60,min_periods=45).std()
shock=(ret.rolling(5,min_periods=5).sum()/(v20*np.sqrt(5)+1e-8))*(v20/(v60+1e-8))
f=(-shock).shift(1).replace([np.inf,-np.inf],np.nan).clip(-5,5)
def calc(h):
 vals=[]; ds=[]; ns=[]
 for i in range(len(prices)-h):
  z=pd.concat([f.iloc[i],prices.iloc[i+h]/prices.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(prices.index[i]);ns.append(len(z))
 x=pd.Series(vals,index=ds); return x,np.mean(ns)
print('universe',len(U),'dates',prices.index.min().date(),prices.index.max().date())
for h in [5,10,20,40,60]:
 x,an=calc(h); print('H',h,'dates',len(x),'avgN',round(an,2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1)*np.sqrt(252),6),'hit',round((x>0).mean(),4))
x,_=calc(10)
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=x[(x.index>=a)&(x.index<=b+'-12-31')]
 if len(q): print('REG',a,b,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),6))
print('coverage',round(f.notna().sum(axis=1).div(len(U)).mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20350607_volshock_reversal_signal.csv',index=False)
