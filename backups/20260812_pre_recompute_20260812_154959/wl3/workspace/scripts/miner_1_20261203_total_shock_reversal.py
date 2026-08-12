import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date');D[s]=x
idx=sorted(set.intersection(*[set(x.index) for x in D.values()])); O=pd.DataFrame({s:D[s].open for s in U}).reindex(idx); C=pd.DataFrame({s:D[s].close for s in U}).reindex(idx); H=pd.DataFrame({s:D[s].high for s in U}).reindex(idx);L=pd.DataFrame({s:D[s].low for s in U}).reindex(idx); y=C.shift(-1)/C-1
gap=O/C.shift(1)-1; intr=C/O-1; rv=C.pct_change().rolling(20,min_periods=12).std()
# A gap-and-close shock reversal: fade the combined overnight and intraday move, with larger weight on unusual (risk-scaled) shocks.
f=(-(gap+intr)/(rv+1e-12)).clip(-5,5)
q=[];ns=[]
for d in f.index:
 a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
 if len(a)>=8 and a.f.nunique()>1:q.append(spearmanr(a.f,a.y).statistic);ns.append(len(a))
q=np.array(q);print('candidate total_shock_reversal cutoff 2026-12-02 dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),6))
for n,lo,hi in [('early','2020-06','2022-12'),('mid','2023-01','2024-12'),('late','2025-01','2026-12')]:
 z=[]
 for d in f.loc[lo:hi].index:
  a=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(a)>=8 and a.f.nunique()>1:z.append(spearmanr(a.f,a.y).statistic)
 z=np.array(z);print('regime',n,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
f.rename_axis('date').to_csv('scripts/miner_1_20261203_total_shock_reversal_signal.csv')
