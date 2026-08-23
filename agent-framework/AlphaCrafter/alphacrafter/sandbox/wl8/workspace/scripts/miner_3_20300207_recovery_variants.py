import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2030-02-06'
P=pd.concat({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A},axis=1).sort_index().loc[:cut]
r=P.pct_change(); raw=-(P.shift(1)/P.rolling(120,min_periods=80).max().shift(1)-1)/r.rolling(20,min_periods=15).std().shift(1); tr=P.shift(1)/P.shift(61)-1
fw=P.shift(-10)/P-1
for name,f in [('raw',raw),('gate',raw*(tr>0)),('boost',raw*(1+4*tr.clip(lower=0))),('soft',raw*(1+tr.clip(lower=0)*10))]:
 out=[]
 for d in P.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna(); c=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
  if np.isfinite(c):out.append((d,c,len(z)))
 R=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); x=R.ic
 print(name,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round(R.n.mean(),2),round((x>0).mean(),4),'r360',round(R.loc['2029-02-06':,'ic'].mean(),6))
