import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2031-12-25'); base=Path('../persistent/stock_data'); ix=Path('../persistent/index_data')
p={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).query('date <= @cut').set_index('date')['close'] for s in U}
p=pd.DataFrame(p).sort_index(); r=p.pct_change(10)
v=pd.read_csv(ix/'VIX.csv',parse_dates=['date']).query('date <= @cut').set_index('date')['close'].reindex(p.index).ffill()
pct=v.rolling(252,min_periods=126).rank(pct=True)
res=r.sub(r.median(axis=1),axis=0)
sig=(-res.rolling(3,min_periods=3).mean()).mul(1+0.75*pct,axis=0).shift(1)
for h in [5,10,20]:
 fwd=p.shift(-h)/p-1; cs=[]; ns=[]
 for d in p.index:
  x=sig.loc[d]; y=fwd.loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8: cs.append(spearmanr(x[ok],y[ok]).statistic); ns.append(ok.sum())
 a=np.asarray(cs); ic=np.nanmean(a); icir=ic/(np.nanstd(a,ddof=1)/np.sqrt(len(a)))
 print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={icir:.8f} hit={np.mean(a>0):.6f} coverage={np.mean(ns)/15:.6f}')
 if h==5:
  for n in [365,730,1095]:
   z=a[-n:]; print(f'recent_{n}: IC={np.nanmean(z):.8f} ICIR={np.nanmean(z)/(np.nanstd(z,ddof=1)/np.sqrt(len(z))):.8f}')
