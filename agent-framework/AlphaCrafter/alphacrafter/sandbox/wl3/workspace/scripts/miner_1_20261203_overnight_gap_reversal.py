import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2026-12-02'; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').sort_values('date').set_index('date'); D[s]=x[['open','close']]
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); op=pd.DataFrame({s:D[s].open for s in U}).reindex(idx); cl=pd.DataFrame({s:D[s].close for s in U}).reindex(idx)
# Overnight gap reversal: fade the open-to-prior-close gap, scaled by recent close volatility.
gap=op/cl.shift(1)-1
vol=cl.pct_change().rolling(20,min_periods=12).std()
f=(-gap/(vol+1e-12)).clip(-5,5)
y=cl.shift(-1)/cl-1
vals=[]; ns=[]
for d in f.index:
 a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
 if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: vals.append(spearmanr(a.f,a.y).statistic); ns.append(len(a))
q=np.array(vals); print('candidate overnight_gap_reversal_scaled cutoff',cut,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for name,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-12')]:
 qq=[]
 for d in f.loc[lo:hi].index:
  a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: qq.append(spearmanr(a.f,a.y).statistic)
 qq=np.array(qq); print('regime',name,len(qq),round(qq.mean(),6),round(qq.mean()/qq.std(ddof=1),6))
for h in [3,5,10]:
 qq=[]
 for d in f.index:
  a=pd.DataFrame({'f':f.loc[d],'y':(cl.shift(-h)/cl-1).loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: qq.append(spearmanr(a.f,a.y).statistic)
 qq=np.array(qq); print('decay',h,len(qq),round(qq.mean(),6),round(qq.mean()/qq.std(ddof=1),6))
f.rename_axis('date').to_csv('scripts/miner_1_20261203_overnight_gap_reversal_signal.csv')
print('period',cl.index.min(),cl.index.max())
