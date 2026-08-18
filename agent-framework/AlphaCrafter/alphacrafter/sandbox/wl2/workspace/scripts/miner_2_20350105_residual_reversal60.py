import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index(); P=P.loc[:'2035-01-04']; r=P.pct_change(); b=r.mean(axis=1)
# 60-session market-beta residual reversal, volatility scaled; all rolling inputs lagged before forecast.
beta=r.rolling(90,min_periods=60).cov(b).div(b.rolling(90,min_periods=60).var(),axis=0).shift(1)
res=r.sub(beta.mul(b,axis=0)); rv=res.rolling(60,min_periods=40).std().shift(1)
sig=-res.rolling(60,min_periods=40).sum().shift(1)/(rv*np.sqrt(60)+1e-9)
rows=[]
for h in [5,10,20,40]:
 y=P.pct_change(h).shift(-h); q=[]; ds=[]; ns=[]
 for d in sig.index:
  ok=sig.loc[d].notna()&y.loc[d].notna()
  if ok.sum()>=8:
   q.append(spearmanr(sig.loc[d][ok],y.loc[d][ok]).statistic);ds.append(d);ns.append(ok.sum())
 q=pd.Series(q,index=ds); print('h',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
 for n,s in [('2026-2029',q.loc['2026':'2029']),('2030-2032',q.loc['2030':'2032']),('2033-2035',q.loc['2033':'2035'])]: print(n,len(s),'IC %.6f ICIR %.6f'%(s.mean(),s.mean()/s.std(ddof=1)))
 rows.append((h,q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.4f turnover %.4f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('../persistent/miner_2_20350105_residual_reversal60_signal.csv',index=False)
print('artifact rows',len(out))
