import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2029-06-13'); fs={}
for a in A:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index();fs[a]=d.close.replace(0,np.nan)
px=pd.concat(fs,axis=1).sort_index().loc[:E].ffill(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.loc[:E].reindex(px.index).ffill()
base=-px.pct_change(60)/(px.pct_change().rolling(20).std()*np.sqrt(252)); gate=(v>v.rolling(60).median()).astype(float); sig=base*gate.replace(0,np.nan)
fw=px.shift(-10)/px-1; z=[]; ds=[];ns=[]
for d in sig.index:
 ok=sig.loc[d].notna()&fw.loc[d].notna()
 if ok.sum()>=8:z.append(spearmanr(sig.loc[d][ok],fw.loc[d][ok]).statistic);ds.append(d);ns.append(ok.sum())
z=np.array(z);print('period',ds[0].date(),ds[-1].date(),'dates',len(z),'mean_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
for c in ['2026-07-16','2027-01-01','2028-06-14','2029-01-01']:
 q=z[np.array(ds)>=pd.Timestamp(c)];print(c,len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
