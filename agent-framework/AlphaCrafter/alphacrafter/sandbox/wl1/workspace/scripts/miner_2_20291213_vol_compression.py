import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-12-12'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change()
# Volatility-compression continuation: medium trend favored when recent volatility is
# compressed versus its long baseline, with cross-sectional volatility normalization.
trend=p/p.shift(40)-1
shortvol=r.rolling(10,min_periods=7).std(); longvol=r.rolling(60,min_periods=30).std()
compression=(longvol/(shortvol+1e-8)).clip(0.5,3.0)
sig=(trend*compression).shift(1)
print('cutoff',cutoff.date(),'assets',len(syms),'dates',len(ix))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.asarray(z); print('H',h,'dates',len(z),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),np.mean(z),np.mean(z)/np.std(z,ddof=1),np.mean(z>0)))
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.index.name='date';sig.to_csv('scripts/miner_2_20291213_vol_compression_signal.csv')
