import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-24')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date').close; px[s]=d
P=pd.DataFrame(px).sort_index(); L=np.log(P).diff(); V=L.rolling(20,min_periods=15).std(); F=-(np.log(P/P.shift(5)))/(V*np.sqrt(5))
# per-asset next-observation return, avoiding weekend/holiday fake alignment
R=pd.DataFrame({s: px[s].pct_change().shift(-1) for s in U})
rows=[]; turns=[]; prev=None
for dt in F.index:
 x=pd.concat([F.loc[dt],R.loc[dt]],axis=1).dropna()
 if len(x)>=8:
  ic=spearmanr(x.iloc[:,0],x.iloc[:,1]).statistic; rows.append((dt,ic,len(x)))
  r=F.loc[dt].rank(pct=True); turns.append(np.mean(abs(r.dropna()-prev.loc[r.dropna().index]))) if prev is not None else None; prev=r
z=np.array([x[1] for x in rows]); print('dates',len(z),'range',rows[0][0],rows[-1][0],'avg_n',np.mean([x[2] for x in rows]),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',np.mean(z>0),'turnover',np.nanmean(turns))
for name,start,end in [('2020-22','2020-01-01','2022-12-31'),('2023','2023-01-01','2023-12-31'),('2024','2024-01-01','2024-12-31'),('2025','2025-01-01','2025-12-31'),('2026+','2026-01-01','2027-02-24')]:
 a=np.array([x[1] for x in rows if pd.Timestamp(start)<=x[0]<=pd.Timestamp(end)]);print(name,len(a),a.mean(),a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
F.stack().rename('signal').reset_index().rename(columns={'level_1':'asset'}).to_csv('../persistent/factor_signals_miner_2_20270225_volnorm_reversal.csv',index=False)
