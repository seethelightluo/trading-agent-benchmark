import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.astype(float) for a in A}
P=pd.DataFrame(P).sort_index().loc[:'2035-07-04']; R=P.pct_change()
# Candidate: contrarian residual return versus the contemporaneous cross-asset median,
# normalized by each asset's 20d realized volatility and lagged one completed day.
mom=P.pct_change(20); resid=mom.sub(mom.median(axis=1),axis=0)
vol=R.rolling(20,min_periods=12).std(); F=(-resid/vol).shift(1)

def evaluate(h):
 rows=[]
 Y=P.shift(-h)/P-1
 for dt in F.index:
  q=pd.concat([F.loc[dt],Y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(ic): rows.append((dt,len(q),ic))
 return pd.DataFrame(rows,columns=['date','n','ic'])
r=evaluate(10); s=r.ic
print('period',r.date.min().date(),r.date.max().date(),'dates',len(r),'avgN',round(r.n.mean(),2),'assets',len(A))
print('full IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
for k in [120,260,520,780]:
 q=s.tail(min(k,len(s))); print('recent',k,'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:
 z=evaluate(h).ic; print('decay',h,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6))
print('coverage',round(F.notna().sum(axis=1).mean()/len(A),4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
os.makedirs('scripts/artifacts',exist_ok=True)
F.to_csv('scripts/artifacts/miner_2_20350705_residual_reversal_20d_signal.csv',index_label='date')
r.to_csv('scripts/artifacts/miner_2_20350705_residual_reversal_20d_ic.csv',index=False)
