import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
end=min(max(x.index.max() for x in D.values()),pd.Timestamp('2026-09-09'))
dates=D['SPX'].index[(D['SPX'].index>='2020-04-01')&(D['SPX'].index<=end)]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U})
r=C.pct_change(); vol=r.rolling(60,min_periods=40).std()
# Mean-reversion signal: recent 5d move, scaled by slower volatility; lower recent winners should outperform.
F=-(C/C.shift(5)-1.0)/(vol*np.sqrt(5)+0.005)
F=F.shift(1); Y=C.pct_change().shift(-1)
q=[]; ns=[]; used=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  q.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); used.append(dt)
q=np.asarray(q); print('idea volatility-scaled 5d reversal; end',end.date())
print('dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
for k in [63,126,252,504]:
 x=q[-k:]; print('recent',k,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
for a,b in [('2020','2021'),('2022','2023'),('2024','2026')]:
 x=q[[str(d)[:4] in ([a,b] if a!='2024' else ['2024','2025','2026']) for d in used]]
 print('regime',a,b,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
print('signal_artifact', 'formula=-(5d_return)/(60d_daily_vol*sqrt5+0.005); lag=1')
