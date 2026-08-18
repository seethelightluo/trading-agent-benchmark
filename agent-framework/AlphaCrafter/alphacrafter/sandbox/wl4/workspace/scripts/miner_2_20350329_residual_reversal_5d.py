import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.concat([pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in symbols],axis=1).sort_index()
r=p.pct_change()
# Cross-asset beta-neutral short-term reversal: remove contemporaneous equal-weight market move
m=r.mean(axis=1)
window=60
betas=pd.concat([(r[s].rolling(window,min_periods=30).cov(m)/m.rolling(window,min_periods=30).var()).rename(s) for s in symbols],axis=1)
resid=r-betas.mul(m,axis=0)
# negative trailing 5d residual return, scaled by trailing residual downside risk; lag one completed day
rv=resid.rolling(20,min_periods=15).std()
sig=(-resid.rolling(5,min_periods=5).sum()/rv.replace(0,np.nan)).shift(1)
fwd=p.shift(-10)/p-1
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=residual_reversal_5d; dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for days in [120,260,520,1040]:
 q=a.tail(days); print('recent',days,'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1),(q.ic>0).mean()))
rank=sig.rank(axis=1,pct=True); print('turnover_proxy',rank.diff().abs().mean(axis=1).dropna().mean())
for h in [1,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC %.8f ICIR %.8f'%(np.nanmean(rr),np.nanmean(rr)/np.nanstd(rr,ddof=1)))
os.makedirs('scripts/artifacts',exist_ok=True); sig.to_csv('scripts/artifacts/miner_2_20350329_residual_reversal_5d_signal.csv'); a.to_csv('scripts/artifacts/miner_2_20350329_residual_reversal_5d_ic.csv')
