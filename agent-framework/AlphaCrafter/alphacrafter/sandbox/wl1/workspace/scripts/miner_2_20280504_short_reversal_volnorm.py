import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2028-05-03')
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().loc[:cut] for s in U};idx=sorted(set().union(*[set(x.index) for x in P.values()]));C=pd.DataFrame({s:x.close.reindex(idx) for s,x in P.items()}).ffill();R=C.pct_change()
# Short-term reversal normalized by 20d volatility, with a liquidity-independent cross-asset signal.
vol=R.rolling(20,min_periods=15).std();f=(-R.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+.002)).shift(1)
print('factor short_reversal_volnorm',len(C),C.index.max().date())
for h in [1,3,5,10]:
 a=[];n=[];ds=[]
 for i in range(len(C)-h):
  q=pd.concat([f.iloc[i].rename('f'),(C.iloc[i+h]/C.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:a.append(spearmanr(q.f,q.y).statistic);n.append(len(q));ds.append(C.index[i])
 a=np.array(a);ds=pd.DatetimeIndex(ds);print('h',h,'dates',len(a),'N',np.mean(n),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('turnover',((f.rank(pct=True)-f.rank(pct=True).shift()).abs().stack().groupby(level=0).mean().dropna().mean()))
f.to_csv('scripts/miner_2_20280504_short_reversal_volnorm_signal.csv',index_label='date')
