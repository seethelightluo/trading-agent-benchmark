import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-11-28'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change();
# Candidate: short-horizon reversal normalized by idiosyncratic volatility, with a slow trend gate.
# Reversal is favored only when the 60d trend is not strongly negative, avoiding falling knives.
rv=-(p/p.shift(3)-1); vol=r.rolling(20,min_periods=10).std(); trend=p/p.shift(60)-1
gate=(1+trend.clip(-.15,.15)/.15).clip(.25,1.75)
sig=(rv/(vol+1e-8)*gate).shift(1)
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print('H',h,'dates',len(z),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),np.mean(z),np.mean(z)/np.std(z,ddof=1),np.mean(z>0)))
 if h==1:
  for n,m in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for d in sig.index[m]:
    ok=sig.loc[d].notna()&f.loc[d].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic)
   q=np.array(q); print(n,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()));sig.index.name='date';sig.to_csv('scripts/miner_2_20291129_gated_reversal_signal.csv')
