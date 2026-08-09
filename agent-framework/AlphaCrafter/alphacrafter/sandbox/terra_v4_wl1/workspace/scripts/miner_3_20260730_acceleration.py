import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
# common dates, factor = recent return acceleration: 5d return minus average 20d return pace
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
f=(P/P.shift(5)-1) - (P/P.shift(20)-1)/4
# Cross sectional IC by date against forward returns
rows=[]; horizons=[1,5,10]
for dt in f.index:
 x=f.loc[dt]
 for h in horizons:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   rows.append((dt,h,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('dates instruments',df[df.h==1].date.nunique(),df[df.h==1].n.mean(),'coverage',df[df.h==1].n.mean()/15)
for h in horizons:
 q=df[df.h==h].ic.dropna(); print('H',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=df[(df.h==1)&(df.date>=a)&(df.date<=b)].ic
 print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan)
# turnover rank top/bottom changes
r=f.rank(axis=1,pct=True); turn=(r-r.shift(1)).abs().mean(axis=1).dropna(); print('turnover',turn.mean())
# correlation with known approximate signal artifacts
for fn in ['miner_1_20260716_peer_median_leadlag_5d.json','miner_2_20260716_risk_adjusted_momentum_20d.json','miner_1_20260716_short_term_reversal_5d.json']:
 try:
  import json
  j=json.load(open('factors/'+fn)); print('library',fn,j.get('validation',{}).get('metrics',{}).get('ic'),j.get('validation',{}).get('metrics',{}).get('max_abs_library_correlation'))
 except: pass
