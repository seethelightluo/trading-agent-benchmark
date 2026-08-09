import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p)
 x['date']=pd.to_datetime(x['date']); x=x[x['date']<=pd.Timestamp('2026-12-17')].sort_values('date').set_index('date')
 D[s]=x['close'].astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index()
# candidate: 60-session trend strength, return divided by realized vol, avoids lookahead
mom=R.rolling(60,min_periods=45).sum(); vol=R.rolling(60,min_periods=45).std()*np.sqrt(60)
F=mom/vol
# align factor at t with next day return
rows=[]
for dt in R.index:
 f=F.loc[dt]; y=R.shift(-1).loc[dt]
 z=pd.concat([f.rename('f'),y.rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for k in [1,5,10]:
 y=R.shift(-k).rolling(k).sum().shift(-(k-1))
 rr=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.f,z.y).statistic)
 q=pd.Series(rr).dropna(); print(k,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'dates',len(q))
print('daily by year',a.groupby(a.index.year).ic.mean().to_dict(),'avg_n',a.n.mean(),'coverage',a.n.mean()/15)
# approximate turnover rank changes
rank=F.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
