import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2034-12-08'].ffill(); r=P.pct_change()
# Volatility-normalized short-term reversal: lagged 5-day loss divided by 20-day risk.
v=r.rolling(20,min_periods=15).std().shift(1)
sig=-(P.pct_change(5).shift(1))/(v*np.sqrt(5)+1e-12)
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); z=[]; ns=[]; ds=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=pd.Series(z,index=ds)
 print('h',h,'dates',len(z),'mean_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 for name,sub in [('2020-2024',z.loc[:'2024-12-31']),('2025-2029',z.loc['2025':'2029-12-31']),('2030-2034',z.loc['2030':])]: print(name,len(sub),round(sub.mean(),6),round(sub.mean()/sub.std(ddof=1),6))
print('coverage',round(sig.notna().mean().mean(),4),'turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_1_20341208_volnorm_reversal_signal.csv',index=False)
