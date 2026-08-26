import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-08-08')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# Relative trend consistency: positive share of daily relative returns over 60d,
# multiplied by relative cumulative trend and lagged one session.
rr=r.sub(r.median(axis=1),axis=0)
cons=rr.gt(0).rolling(60,min_periods=45).mean()-0.5
mom=p.pct_change(60).sub(p.pct_change(60).median(axis=1),axis=0)
f=(cons*mom).shift(1)
rowsall={}
for h in [10,20,40,60]:
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): rows.append((dt,q,len(z)))
 a=pd.DataFrame(rows,columns=['date','ic','n']); rowsall[h]=a
 ic=a.ic.mean(); ir=ic/(a.ic.std(ddof=1)+1e-12)
 print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('2024-2026',a[(a.date>='2024-01-01')&(a.date<='2026-12-31')]),('2027-2030',a[a.date>='2027-01-01'])]:
  if len(sl): print('regime',h,name,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
rank=f.rank(axis=1,pct=True); print('turnover_proxy',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
f.index.name='date'; f.to_csv('scripts/miner_1_20300808_relative_trend_consistency_60d_signal.csv')
