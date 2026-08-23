import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'; cutoff=pd.Timestamp('2026-09-09')
px=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cutoff]
r=px.pct_change(); d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change().reindex(r.index)
beta=pd.DataFrame({s:r[s].rolling(60,min_periods=45).cov(d)/d.rolling(60,min_periods=45).var() for s in U})
shock=d.rolling(5,min_periods=5).sum(); f=-beta.mul(shock,axis=0)
def run(h):
 out=[]
 y=r.shift(-1).rolling(h).sum().shift(-(h-1))
 for dt in r.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); return a
for h in [1,5,10]:
 a=run(h); print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('regime',lo,hi,'dates',len(q),'ICIR',round(q.mean()/q.std(ddof=1),6))
ranks=f.rank(axis=1,pct=True); print('turnover',round(ranks.diff().abs().mean(axis=1).dropna().mean(),6))
