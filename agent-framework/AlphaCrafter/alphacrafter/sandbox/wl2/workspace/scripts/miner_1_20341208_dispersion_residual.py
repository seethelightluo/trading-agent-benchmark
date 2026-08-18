import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index().loc[:'2034-12-08'].ffill(); r=P.pct_change(); cs=r.sub(r.median(axis=1),axis=0); v=r.rolling(20,min_periods=15).std().shift(1)
# Apply residual reversal only during high cross-sectional dispersion; otherwise neutral (rank ties handled by valid subset).
disp=r.std(axis=1).rolling(60,min_periods=40).mean(); thresh=disp.rolling(252,min_periods=126).quantile(.65).shift(1)
sig=(-cs.shift(1)/(v+1e-12)).where(disp>thresh)
f=P.pct_change(1).shift(-1);z=[];ds=[];ns=[]
for d in sig.index:
 ok=sig.loc[d].notna()&f.loc[d].notna()
 if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ds.append(d);ns.append(ok.sum())
z=pd.Series(z,index=ds)
print('dates',len(z),'mean_n',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for name,sub in [('2020-2024',z.loc[:'2024-12-31']),('2025-2029',z.loc['2025':'2029-12-31']),('2030-2034',z.loc['2030':])]:print(name,len(sub),sub.mean(),sub.mean()/sub.std(ddof=1))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_1_20341208_dispersion_residual_signal.csv',index=False)
