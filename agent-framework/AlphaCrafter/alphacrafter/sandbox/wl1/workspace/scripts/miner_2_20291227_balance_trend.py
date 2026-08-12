import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-12-26')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U}
idx=sorted(set().union(*[set(x.index) for x in P.values()])); px=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill(); r=px.pct_change()
# Volatility-balanced defensive carry: downside/upside semivolatility balance plus medium trend,
# with cross-sectional ranking to reduce scale differences across asset classes.
up=r.clip(lower=0).rolling(40,min_periods=25).std(); dn=(-r.clip(upper=0)).rolling(40,min_periods=25).std()
balance=np.log((up+1e-8)/(dn+1e-8)); trend=px.pct_change(20)
f=(balance.rank(axis=1,pct=True)+0.5*trend.rank(axis=1,pct=True)).shift(1)
f.index.name='date'; f.to_csv('scripts/miner_2_20291227_balance_trend_signal.csv')
for h in [1,5,10,20]:
 I=[]; ds=[]; ns=[]
 for i in range(len(px)-h):
  q=pd.concat([f.iloc[i].rename('f'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:I.append(spearmanr(q.f,q.y).statistic);ds.append(px.index[i]);ns.append(len(q))
 a=np.array(I); d=pd.DatetimeIndex(ds)
 print('H',h,'dates',len(a),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
 if h==1:
  for lab,m in [('2020-25',(d<pd.Timestamp('2026-01-01'))),('2026+',d>=pd.Timestamp('2026-01-01')),('2028+',d>=pd.Timestamp('2028-01-01')),('2029YTD',d>=pd.Timestamp('2029-01-01'))]:
   x=a[m]; print(lab,'dates',len(x),'IC %.6f ICIR %.6f'%(x.mean(),x.mean()/x.std(ddof=1)))
print('coverage %.6f turnover %.6f dates %d assets %d'%(f.notna().mean().mean(),f.rank(pct=True).diff().abs().mean(axis=1).dropna().mean(),len(px),len(U)))
