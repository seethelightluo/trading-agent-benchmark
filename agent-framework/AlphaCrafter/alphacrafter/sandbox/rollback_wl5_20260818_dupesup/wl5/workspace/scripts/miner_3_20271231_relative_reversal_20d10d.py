import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
C=pd.DataFrame({s:D[s]['close'] for s in U})
cut=pd.Timestamp('2027-12-31')
for lb,h in [(20,10),(20,5),(10,10),(5,10)]:
 r=C.pct_change(lb); f=-(r.sub(r.median(axis=1),axis=0)); Y=C.shift(-h)/C-1
 vals=[]; dates=[]; ns=[]
 for d in f.index:
  d=pd.Timestamp(d)
  if d>cut: continue
  g=pd.DataFrame({'f':f.loc[d],'y':Y.loc[d]}).replace([np.inf,-np.inf],np.nan).dropna()
  if d>=pd.Timestamp('2020-01-01') and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:
   vals.append(spearmanr(g.f,g.y).statistic); dates.append(d); ns.append(len(g))
 z=np.asarray(vals); dates=pd.DatetimeIndex(dates); online=z[dates>=pd.Timestamp('2026-07-16')]; recent=z[dates>=pd.Timestamp('2027-01-01')]
 print(f'lb={lb} h={h} dates={len(z)} avg_n={np.mean(ns):.2f} coverage={np.mean(ns)/15:.4f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={np.mean(z>0):.4f} online_dates={len(online)} online_IC={online.mean():.6f} online_ICIR={online.mean()/online.std(ddof=1):.6f} recent_IC={recent.mean():.6f} recent_ICIR={recent.mean()/recent.std(ddof=1):.6f}')
 if (lb,h)==(20,10): pd.DataFrame({'date':dates,'ic':vals,'n':ns}).to_csv('scripts/miner_3_20271231_relative_reversal_20d10d_ic.csv',index=False)
