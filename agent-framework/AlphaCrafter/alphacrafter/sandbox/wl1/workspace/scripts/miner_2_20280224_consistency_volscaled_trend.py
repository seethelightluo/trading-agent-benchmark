import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2028-02-23')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U};idx=sorted(set().union(*[set(x.index) for x in P.values()]));C=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill();R=C.pct_change()
# Long horizon trend, penalized by realized volatility and instability of daily direction.
trend=R.rolling(60,min_periods=45).sum(); vol=R.rolling(60,min_periods=45).std()*np.sqrt(60); consistency=R.rolling(60,min_periods=45).mean().abs()/(R.abs().rolling(60,min_periods=45).mean()+1e-8)
f=(trend/(vol+0.004)*consistency).shift(1)
print('idea consistency_volscaled_trend universe',len(U),'dates',len(C),'cutoff',C.index.max().date())
for h in [10,20]:
 a=[];n=[];ds=[]
 for i in range(len(C)-h):
  q=pd.concat([f.iloc[i].rename('f'),(C.iloc[i+h]/C.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:a.append(spearmanr(q.f,q.y).statistic);n.append(len(q));ds.append(C.index[i])
 a=np.array(a);ds=pd.DatetimeIndex(ds);print('h',h,'valid_dates',len(a),'avgN',round(np.mean(n),2),'coverage',round(np.mean(n)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
 for lab,m in [('2026+',ds>=pd.Timestamp('2026-01-01')),('2027+',ds>=pd.Timestamp('2027-01-01')),('2028+',ds>=pd.Timestamp('2028-01-01'))]:
  z=a[m];print(lab,'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'dates',int(m.sum()))
rk=f.rank(axis=1,pct=True);print('turnover',round((rk-rk.shift()).abs().stack().groupby(level=0).mean().dropna().mean(),6));f.to_csv('scripts/miner_2_20280224_consistency_volscaled_trend_signal.csv',index_label='date')
