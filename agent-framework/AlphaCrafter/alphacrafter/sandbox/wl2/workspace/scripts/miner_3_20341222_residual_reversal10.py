import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); r=P.pct_change(); bench=r.mean(axis=1)
# Cross-asset residual reversal: remove each asset's rolling 60d beta to the equal-weight world return,
# then fade the lagged 10d residual move, scaled by residual volatility.
beta=r.rolling(60,min_periods=40).cov(bench).div(bench.rolling(60,min_periods=40).var(),axis=0).shift(1)
res=r.sub(beta.mul(bench,axis=0)); rv=res.rolling(30,min_periods=20).std().shift(1)
sig=-res.rolling(10,min_periods=8).sum().shift(1)/(rv*np.sqrt(10)+1e-9)
for h in [1,5,10,20]:
 f=P.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for d in sig.index:
  x,y=sig.loc[d],f.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum()); ds.append(d)
 z=pd.Series(vals,index=ds); print('h',h,'dates',len(z),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 for q,s in [('early',z.loc[:'2025']),('mid',z.loc['2026':'2030']),('recent',z.loc['2031':])]: print(q,len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
print('coverage %.4f turnover %.4f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_3_20341222_residual_reversal10_signal.csv',index=False)
