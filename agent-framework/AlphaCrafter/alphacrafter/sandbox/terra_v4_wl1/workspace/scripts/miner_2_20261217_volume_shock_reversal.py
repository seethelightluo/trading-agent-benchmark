import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:cut]; V=pd.DataFrame({s:D[s].volume for s in U}).reindex(P.index)
r=P.pct_change(); vs=V.div(V.rolling(20,min_periods=15).mean()).replace([np.inf,-np.inf],np.nan)
# Continuous volume-shock fade of lagged 3-day return; volume surprise is lagged before use.
f=-(P/P.shift(3)-1)*np.log1p(vs.shift(1).clip(lower=0))
for h in [1,5,10]:
 y=P.shift(-h).div(P)-1; rows=[]
 for d in P.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>=3: rows.append((d,spearmanr(q.f,q.y).statistic,len(q)))
 a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=a.ic
 print('H',h,'dates',len(ic),'avg_n',round(a.n.mean(),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4))
 if h==1:
  for lab,g in a.ic.groupby(pd.cut(a.index,[pd.Timestamp('2019-12-31'),pd.Timestamp('2022-12-31'),pd.Timestamp('2024-12-31'),cut],labels=['2020-22','2023-24','2025-26'])): print('REG',lab,'dates',len(g),'IC',round(g.mean(),6),'ICIR',round(g.mean()/g.std(ddof=1),6))
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),4),'period',P.index.min().date(),P.index.max().date())
