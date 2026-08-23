import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-03-08')
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}).sort_index();P=P[P.index<=end]
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(P.index).ffill(); vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std(); shock=v.pct_change(3).shift(1).clip(lower=0,upper=.5)
# lagged 1-day reversal, amplified only after positive VIX shock; all inputs through prior date
f=-(P.shift(1)/P.shift(2)-1).mul(1+2*shock,axis=0).replace([np.inf,-np.inf],np.nan); fw=P.shift(-1)/P-1
rows=[]
for d in P.index:
 x,y=f.loc[d],fw.loc[d];ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('candidate VIX-positive-shock amplified 1d reversal');print('dates',len(z),'avg_n',z.n.mean(),'coverage',z.n.sum()/(len(z)*15));print('IC %.6f ICIR %.6f hit %.4f'%(z.ic.mean(),z.ic.mean()/z.ic.std(),(z.ic>0).mean()))
for n,m in [('2020-22',z.index<'2023-01-01'),('2023-25',(z.index>='2023-01-01')&(z.index<'2026-01-01')),('2026',(z.index>='2026-01-01')&(z.index<'2027-01-01')),('2027+',z.index>='2027-01-01'),('recent90',z.index>=end-pd.Timedelta(days=90))]:
 q=z[m];print(n,len(q),'%.6f %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()) if len(q) else '')
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20280309_vix_shock_reversal_signal.csv',index=False)
