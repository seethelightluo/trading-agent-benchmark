import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data'
px={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
prices=pd.DataFrame(px).sort_index().loc[:'2035-06-06']; r=prices.pct_change()
ret20=prices/prices.shift(20)-1; vol40=r.rolling(40,min_periods=30).std()*np.sqrt(252)
csdisp=r.apply(lambda x:x.std(),axis=1); disp=csdisp.rolling(20,min_periods=15).mean(); disp_med=disp.rolling(252,min_periods=126).median()
mult=(1+((disp/disp_med).clip(.5,2)-1)*0.75).to_numpy()[:,None]
f=((-ret20/(vol40+1e-8))*mult).shift(1).clip(-3,3)
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
valid=f.notna().sum(axis=1)/len(U); print('coverage',round(valid.mean(),4),'turnover_proxy',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20350607_dispersion_conditioned_reversal_20d_signal.csv',index=False)
