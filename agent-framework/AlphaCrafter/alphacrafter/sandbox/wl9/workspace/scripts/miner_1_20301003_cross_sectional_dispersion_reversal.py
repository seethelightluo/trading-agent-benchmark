import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None and len(d):
  q=d[['date','close']].copy(); q.date=pd.to_datetime(q.date); q=q.drop_duplicates('date').set_index('date').sort_index(); data[s]=q.close.astype(float)
pd_=pd.DataFrame(data).sort_index(); r=pd_.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Reversal of recent idiosyncratic (cross-sectional demeaned) shocks, scaled by own risk,
# activated when cross-sectional return dispersion is elevated; lagged one session.
ret10=pd_.pct_change(10); resid=ret10.sub(ret10.median(axis=1),axis=0)
disp=ret10.std(axis=1).rolling(20,min_periods=15).rank(pct=True)
f=(-resid/vol).where(disp.ge(0.5), -resid/vol*0.5).shift(1)
for h in [5,10,20,40,60]:
 fr=pd_.shift(-h).div(pd_)-1; vals=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); ir=ic/a.std(ddof=1)*np.sqrt(252)
 print(f'H={h} dates={len(a)} avgN={np.mean(ns):.2f} coverage={np.mean(ns)/len(U):.4f} IC={ic:.6f} ICIR={ir:.6f} hit={np.mean(a>0):.4f}')
 if h==20:
  for lo,hi,nm in [('2024-01-01','2026-12-31','2024-26'),('2027-01-01','2029-12-31','2027-29'),('2030-01-01','2030-12-31','2030YTD')]:
   q=a[(a.index>=lo)&(a.index<=hi)]; print(f' regime={nm} dates={len(q)} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan:.6f}')
print(f'turnover_proxy={f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} instruments={len(data)} dates={len(pd_)}')
f.to_csv('scripts/miner_1_20301003_cross_sectional_dispersion_reversal_signal.csv',index_label='date')
