import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cut=pd.Timestamp('2030-01-09'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cut] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change()
# Breakout quality: proximity to 120d high combined with positive 20d return, normalized by volatility.
dd=p/p.rolling(120,min_periods=80).max()-1; ret20=p/p.shift(20)-1; vol40=r.rolling(40,min_periods=25).std()*np.sqrt(252)
sig=((1+dd)*ret20/(1+2*vol40)).shift(1); sig.to_csv('scripts/miner_2_20300110_breakout_quality_signal.csv')
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum())
 z=np.array(z);print(f'H {h} dates {len(z)} avgN {np.mean(ns):.2f} IC {z.mean():.6f} ICIR {z.mean()/z.std(ddof=1):.6f} hit {np.mean(z>0):.4f}')
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
