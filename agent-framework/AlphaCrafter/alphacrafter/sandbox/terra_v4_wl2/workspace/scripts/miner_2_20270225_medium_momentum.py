import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in U:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date')
 px[a]=d.set_index('date')['close']
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# candidate: medium-term momentum excluding most recent 5d, volatility-adjusted 60d return
for name,f in {
 'mom60_ex5': R.rolling(60).sum().shift(5),
 'riskadj_mom60_ex5': R.rolling(60).sum().shift(5)/R.rolling(60).std().shift(5),
 'mom20_ex3': R.rolling(20).sum().shift(3)
}.items():
  vals=[]; ics=[]
  for i in range(len(P)-1):
   s=f.iloc[i]; y=R.iloc[i+1]; z=pd.concat([s,y],axis=1).dropna()
   if len(z)>=8:
    vals.append((len(z),s.name)); ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  q=np.array(ics); print(name,'dates',len(q),'avgN',np.mean([v[0] for v in vals]),'IC',np.nanmean(q),'ICIR',np.nanmean(q)/np.nanstd(q,ddof=1),'hit',np.mean(q>0))
  # regimes
  for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
   a=[v for v,(n,d) in zip(q,vals) if lo<=str(d)[:4]<=hi]; print(lo,round(np.nanmean(a),4) if a else None,end='; ')
  print()
