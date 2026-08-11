import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:'2026-07-15'] for s in U}
# CLV reversal strengthened by medium-term trend: daily cross-sectional rank blend,
# lagged one completed observation and independently aligned forward returns.
clv={s:-(2*(D[s].close-D[s].low)/(D[s].high-D[s].low).replace(0,np.nan)) for s in U}
mom={s:D[s].close.pct_change(20) for s in U}
F=pd.DataFrame({s:(clv[s].rank(pct=True)+0.35*mom[s].rank(pct=True)).shift(1) for s in U}).sort_index()
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].close.shift(-h)/D[s].close-1 for s in U}).sort_index(); q=[]; ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z))
 q=np.asarray(q); print('horizon',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 if h==1:
  for yr in range(2020,2027):
   x=[]
   for dt in F.loc[str(yr)].index:
    z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
    if len(z)>=8:x.append(spearmanr(z.f,z.y).statistic)
   print('regime',yr,'dates',len(x),'IC',round(np.mean(x),6) if x else None,'ICIR',round(np.mean(x)/np.std(x,ddof=1),4) if len(x)>1 else None)
print('coverage',round(F.notna().sum().sum()/F.size,4),'rank_turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
