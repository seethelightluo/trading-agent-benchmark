import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
cutoff=pd.Timestamp('2029-12-26')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b=Path('../persistent/stock_data')
cs={s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in syms}
ix=sorted(set().union(*[x.index for x in cs.values()]))
p=pd.DataFrame({s:cs[s].reindex(ix) for s in syms}); r=p.pct_change()
# Shock-reversal within durable trend: fade the latest 5d cross-sectional shock,
# but only where 40d trend agrees with the opposite direction; scale by downside/upside volatility.
ret5=p/p.shift(5)-1; tr40=p/p.shift(40)-1
med5=ret5.median(axis=1); shock=ret5.sub(med5,axis=0)
vol20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# reversal is strongest for negative shock in positive trend and vice versa, normalized for risk
sig=(-shock/(1+3*vol20) * np.sign(tr40)).shift(1)
# retain broad coverage; sign gate is continuous and does not drop names
sig.index.name='date'; sig.to_csv('scripts/miner_2_20291227_shock_reversal_signal.csv')
for h in [1,5,10,20]:
 f=p.shift(-h)/p-1; z=[]; ns=[]; ds=[]
 for d in ix:
  ok=sig.loc[d].notna()&f.loc[d].notna()
  if ok.sum()>=8:
   z.append(spearmanr(sig.loc[d,ok],f.loc[d,ok]).statistic); ns.append(ok.sum()); ds.append(d)
 z=np.asarray(z)
 print(f'H {h} dates {len(z)} avgN {np.mean(ns):.2f} IC {np.mean(z):.6f} ICIR {np.mean(z)/np.std(z,ddof=1):.6f} hit {np.mean(z>0):.4f}')
 if h==1:
  for name,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026+','2026-01-01','2027-12-31'),('2028+','2028-01-01','2028-12-31'),('2029YTD','2029-01-01','2029-12-26')]:
   q=z[np.array([(d>=pd.Timestamp(lo))&(d<=pd.Timestamp(hi)) for d in ds])]
   print(name,'dates',len(q),'IC %.6f ICIR %.6f'%(np.mean(q),np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan))
print('coverage %.6f turnover %.6f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
