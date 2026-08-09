import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-10-08')
C={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for a in A}
r=pd.concat({a:C[a].close.pct_change() for a in A},axis=1)
# volatility shock: inverse ratio of short realized vol to long realized vol; lower recent vol hypothesized stronger forward returns
v5=r.rolling(5,min_periods=4).std();v20=r.rolling(20,min_periods=15).std(); fac=-(v5/v20-1)
cl=pd.concat({a:C[a].close for a in A},axis=1)
for h in [1,5,10]:
 yall=cl.pct_change(h).shift(-h); vals=[];ds=[];nn=[]
 for dt in fac.index:
  x=fac.loc[dt].dropna();y=yall.loc[dt].reindex(x.index).dropna();x=x.reindex(y.index)
  if len(x)>=8 and x.nunique()>1 and y.nunique()>1:vals.append(spearmanr(x,y).statistic);ds.append(dt);nn.append(len(x))
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN',np.mean(nn),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(),(s>0).mean()))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   z=s[(s.index.year>=lo)&(s.index.year<=hi)];print('REG',lo,hi,len(z),z.mean(),z.mean()/z.std())
print('coverage',fac.notna().sum(axis=1).mean()/15,'turnover',fac.rank(pct=True).diff().abs().mean(axis=1).mean())
fac.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261008_volshock_signal.csv',index=False)
