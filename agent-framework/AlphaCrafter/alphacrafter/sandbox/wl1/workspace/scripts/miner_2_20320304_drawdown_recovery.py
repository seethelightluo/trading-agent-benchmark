import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,days=1800)
 if d is None or len(d)<120: d=get_index_daily_data(s,days=1800)
 if d is not None: rows.append(d[['date','close']].assign(symbol=s))
pd_=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill(); r=np.log(pd_).diff()
# Long-horizon drawdown recovery: contrarian recent shock, stronger when 40d drawdown is deep; lagged inputs.
ret10=pd_.pct_change(10); ret40=pd_.pct_change(40); vol20=r.rolling(20).std()
gate=np.maximum(-ret40,0)
f=((-ret10/vol20)*(1+gate)).shift(1)
for h in [1,5,10,20]:
 y=pd_.shift(-h)/pd_-1; vals=[]; ns=[]; turns=[]; prev=None
 for i in range(len(pd_)-h):
  z=f.iloc[i]; q=y.iloc[i]; ok=z.notna()&q.notna()
  if ok.sum()>=8:
   vals.append(z[ok].corr(q[ok])); ns.append(ok.sum()); rank=z.rank(pct=True); turns.append(np.nan if prev is None else (rank-prev).abs().mean()); prev=rank
 a=np.array([x for x in vals if np.isfinite(x)])
 print('horizon',h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,5),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'turn',round(np.nanmean(turns),5))
 if h==10:
  for aa,bb in [('2024-01-01','2026-12-31'),('2027-01-01','2029-12-31'),('2030-01-01','2032-03-04')]:
   q=[]
   for dt in pd_.loc[aa:bb].index:
    ok=f.loc[dt].notna()&y.loc[dt].notna()
    if ok.sum()>=8: q.append(f.loc[dt][ok].corr(y.loc[dt][ok]))
   q=np.array([x for x in q if np.isfinite(x)]); print('regime',aa[:4],len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20320304_drawdown_recovery_signal.csv',index=False)
