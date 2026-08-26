import numpy as np, pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-09-24'); base=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(base/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
R=np.log(P).diff(); vol=R.rolling(20,min_periods=15).std().shift(1)
# Candidate: short-horizon residual reversal, volatility scaled, activated only when cross-sectional dispersion is elevated.
short=np.log(P/P.shift(5)); residual=short.sub(short.median(axis=1),axis=0)
disp=R.rolling(20,min_periods=15).std().mean(axis=1)
threshold=disp.rolling(252,min_periods=120).quantile(.60).shift(1)
gate=(disp>threshold).astype(float)
sig=(-residual/vol).mul(gate,axis=0).replace([np.inf,-np.inf],np.nan)
# use un-gated values as NaN in low dispersion to make conditional factor explicit
fwd={h:np.log(P.shift(-h)/P) for h in [5,10,20,40]}
for h,yy in fwd.items():
 rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 if len(r):
  print('h',h,'dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4),'IC',round(r.ic.mean(),6),'ICIR',round(r.ic.mean()/r.ic.std(ddof=1),6),'hit',round((r.ic>0).mean(),4))
  for label,sub in [('2025_26',r.loc['2025':'2026']),('2027_28',r.loc['2027':'2028']),('recent',r.loc['2028-09-01':])]:
   if len(sub): print(label,'dates',len(sub),'IC',round(sub.ic.mean(),6),'ICIR',round(sub.ic.mean()/sub.ic.std(ddof=1),6))
rank=sig.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6),'data_dates',len(P),'active_dates',int(gate.sum()),'active_rate',round(gate.mean(),4))
sig.stack().rename('signal').to_csv('scripts/miner_1_20290924_dispersion_gated_reversal_signal.csv')
# save 10d IC artifact
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd[10].loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_1_20290924_dispersion_gated_reversal_ic.csv',index=False)
