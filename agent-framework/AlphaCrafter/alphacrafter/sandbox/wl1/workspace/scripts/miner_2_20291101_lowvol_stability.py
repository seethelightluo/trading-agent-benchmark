import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-10-31'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change(); v=r.rolling(20,min_periods=5).std(); breadth=r.gt(0).rolling(10,min_periods=5).mean().mean(axis=1)
# Stability factor: low recent volatility, rewarded more when market breadth is weak (defensive regime).
sig=(1/(v+1e-8)).mul(1+(.5-breadth).clip(-.3,.3),axis=0).shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print('H',h,'dates',len(z),'IC %.6f ICIR %.6f avgN %.2f hit %.4f'%(np.mean(z),np.mean(z)/np.std(z,ddof=1),np.mean(ns),np.mean(z>0)))
 if h==1:
  for n,m in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for d in sig.index[m]:
    ok=sig.loc[d].notna()&f.loc[d].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic)
   q=np.array(q);print(n,'dates',len(q),'IC %.6f ICIR %.6f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1)))
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()));sig.index.name='date';sig.to_csv('scripts/miner_2_20291101_lowvol_stability_signal.csv')
