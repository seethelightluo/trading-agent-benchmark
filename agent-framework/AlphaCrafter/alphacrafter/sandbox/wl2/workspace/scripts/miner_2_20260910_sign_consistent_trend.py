import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
dates=D['SPX'].index[(D['SPX'].index>='2020-01-01')&(D['SPX'].index<='2026-07-15')]
C=pd.DataFrame({s:D[s].close.reindex(dates) for s in U})
r=C.pct_change()
# Sign-consistent trend: average ranks of 5/20/60d returns, weighted toward medium term,
# multiplied by agreement of return signs; lag one completed day.
m5=C.pct_change(5); m20=C.pct_change(20); m60=C.pct_change(60)
base=(m5.rank(axis=1,pct=True)+2*m20.rank(axis=1,pct=True)+m60.rank(axis=1,pct=True))/4
agree=(np.sign(m5)+np.sign(m20)+np.sign(m60)).abs()/3
F=(base*agree).shift(1)
Y=r.shift(-1); q=[]; ns=[]; dates_used=[]
for dt in dates:
 z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
 if len(z)>=8:
  q.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates_used.append(dt)
q=np.asarray(q)
print('factor sign_consistent_multihorizon_trend')
print('dates',len(q),'first',dates_used[0].date(),'last',dates_used[-1].date(),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for k in [63,126,252,504]:
 x=q[-k:]; print('recent',k,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for a,b in [('2020-01-01','2021-12-31'),('2022-01-01','2023-12-31'),('2024-01-01','2026-07-15')]:
 x=q[[a<=str(d.date())<=b for d in dates_used]]; print(a,b,'n',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
