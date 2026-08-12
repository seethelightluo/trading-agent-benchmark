import numpy as np,pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2029-11-14'); base=Path('../persistent/stock_data')
raw={s:pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index().loc[:cutoff] for s in U}; p=pd.DataFrame(raw).sort_index(); r=p.pct_change(); mom=p.pct_change(20); down=r.where(r<0).rolling(40,min_periods=15).std()
# Cross-sectional relative strength normalized by each asset downside risk; lagged one day.
sig=(mom.sub(mom.median(axis=1),axis=0)/(down*np.sqrt(252)+.02)).shift(1)
for h in [1,5,10,20]:
 z=[]; ns=[]; f=p.pct_change(h).shift(-h)
 for dt in sig.index:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:z.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic);ns.append(ok.sum())
 z=np.array(z); print(f'H={h} dates={len(z)} avg_n={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.3f}')
 if h==1:
  for label,mask in [('2020-25',sig.index<'2026-01-01'),('2026+',sig.index>='2026-01-01'),('2028+',sig.index>='2028-01-01'),('2029YTD',sig.index>='2029-01-01')]:
   q=[]
   for dt in sig.index[mask]:
    ok=sig.loc[dt].notna()&f.loc[dt].notna()
    if ok.sum()>=8:q.append(spearmanr(sig.loc[dt,ok],f.loc[dt,ok]).statistic)
   q=np.array(q);print(label,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)))
print('coverage',sig.notna().mean().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.index.name='date';sig.to_csv('scripts/miner_2_20291115_relative_strength_signal.csv')
