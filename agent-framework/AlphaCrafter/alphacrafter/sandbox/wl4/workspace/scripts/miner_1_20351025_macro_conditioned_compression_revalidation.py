import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-10-24']; R=P.pct_change(); base=-(R.rolling(10,min_periods=8).std()/R.rolling(60,min_periods=40).std())
M=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill(); med=M.rolling(60,min_periods=40).median(); z=(M-med)/med
F=base.mul(1+0.5*z.clip(-1,1),axis=0).shift(1); fr=P.shift(-10)/P-1
rows=[]
for dt in F.index:
 q=pd.DataFrame({'f':F.loc[dt],'r':fr.loc[dt]}).dropna()
 if len(q)>=8:
  ic=spearmanr(q.f,q.r).statistic
  if np.isfinite(ic): rows.append((dt,len(q),ic))
r=pd.DataFrame(rows,columns=['date','n','ic']); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean().mean(),4))
os.makedirs('scripts/artifacts',exist_ok=True); r.to_csv('scripts/artifacts/miner_1_20351025_macro_conditioned_compression_ic.csv',index=False); F.to_csv('scripts/artifacts/miner_1_20351025_macro_conditioned_compression_signal.csv')
