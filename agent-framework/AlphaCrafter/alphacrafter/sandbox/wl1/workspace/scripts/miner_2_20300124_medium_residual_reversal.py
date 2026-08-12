import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2030-01-23'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}; ix=sorted(set().union(*[x.index for x in cs.values()])); p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change()
# Medium-term cross-sectional mean reversion: fade each asset's 10d return
# relative to the daily universe median, scaled by 40d realized volatility.
r10=p/p.shift(10)-1; residual=r10.sub(r10.median(axis=1),axis=0); vol40=r.rolling(40,min_periods=25).std()*np.sqrt(252); sig=(-residual/(1+vol40)).shift(1); sig.to_csv('scripts/miner_2_20300124_medium_residual_reversal_signal.csv')
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[];ns=[];ds=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic);ns.append(ok.sum());ds.append(d)
 z=np.asarray(z); print(f'H {h} dates {len(z)} avgN {np.mean(ns):.2f} IC {z.mean():.6f} ICIR {z.mean()/z.std(ddof=1):.6f} hit {np.mean(z>0):.4f}')
 if h==1:
  for name,lo,hi in [('2026-27','2026-01-01','2027-12-31'),('2028','2028-01-01','2028-12-31'),('2029','2029-01-01','2029-12-31'),('2030','2030-01-01','2030-01-23')]:
   q=z[np.array([(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi)) for d in ds])]; print(name,len(q),np.mean(q) if len(q) else np.nan,(np.mean(q)/np.std(q,ddof=1)) if len(q)>1 else np.nan)
