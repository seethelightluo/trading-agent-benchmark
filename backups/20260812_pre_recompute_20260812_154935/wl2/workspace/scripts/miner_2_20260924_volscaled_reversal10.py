import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
# Use only information available through the last completed date in the research window.
last=min(pd.Timestamp('2026-09-23'), max(x.index.max() for x in D.values()))
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<=last)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U}); r=C.pct_change()
# Ten-session horizon reversal, volatility normalized and lagged one session.
# This cadence-aligned signal avoids mixing raw cross-asset volatility scales.
vol=r.rolling(20,min_periods=15).std()*np.sqrt(10)
F=(-r.rolling(10,min_periods=10).sum()/vol.replace(0,np.nan)).shift(1)
Y=C.pct_change(10).shift(-10)
ics=[]; ns=[]; ds=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
q=np.asarray(ics); print('factor vol_scaled_reversal_10d'); print('dates',len(q),'first',ds[0].date(),'last',ds[-1].date(),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4)); print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:
 x=q[-k:];print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2026-09-23')]:
 x=q[[a<=str(d.date())<=b for d in ds]];print(a,b,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
