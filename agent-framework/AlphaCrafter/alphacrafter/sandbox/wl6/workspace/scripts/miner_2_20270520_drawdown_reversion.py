import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# drawdown from trailing 60-session high, lagged one day; more negative = deeper drawdown
raw=P/P.rolling(60,min_periods=40).max()-1
F=-raw.shift(1) # positive for deeper drawdown, hypothesized rebound
# forward 10-day returns, aligned at t
R=P.shift(-10)/P-1
ics=[]; turns=[]; cov=[]; counts=[]
prev=None
for dt in F.index:
 x=F.loc[dt]; y=R.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  ics.append(spearmanr(x[ok],y[ok]).statistic); counts.append(ok.sum()); cov.append(ok.mean())
  ranks=x.rank(pct=True); turns.append(np.mean(np.abs(ranks-(prev if prev is not None else ranks))))
  prev=ranks
z=np.array(ics); print('factor=lagged 60d-high drawdown reversion; horizon=10d')
print('dates',len(z),'avg_instruments',np.mean(counts),'coverage',np.mean(cov),'IC',np.mean(z),'ICIR',np.mean(z)/(np.std(z,ddof=1)/np.sqrt(len(z))),'hit',np.mean(z>0),'turnover',np.mean(turns))
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027YTD','2027-01-01','2027-05-19')]:
 q=(F.index>=a)&(F.index<=b); zz=z[np.array([((F.index[i]>=a)&(F.index[i]<=b)) for i in range(len(F.index))]) if False else []]
 vals=[]
 for dt in F.index[(F.index>=a)&(F.index<=b)]:
  x=F.loc[dt]; y=R.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: vals.append(spearmanr(x[ok],y[ok]).statistic)
 vals=np.array(vals)
 print(name,'dates',len(vals),'IC',np.mean(vals) if len(vals) else np.nan,'ICIR',np.mean(vals)/(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else np.nan)
# save recoverable signal artifact
out=pd.DataFrame(F).reset_index(); out.to_csv('scripts/miner_2_20270520_drawdown_reversion_signal.csv',index=False)
