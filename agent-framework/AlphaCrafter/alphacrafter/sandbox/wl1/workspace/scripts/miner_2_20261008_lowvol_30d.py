import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut='2026-07-15'
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); r=P.pct_change()
# low-volatility risk factor, inverse recent realized volatility, cross-sectionally standardized by rank
f=-r.rolling(30,min_periods=20).std()
for h in [5,10,20]:
 a=[]
 for dt in P.loc[:cut].index:
  q=pd.concat([f.loc[dt],P.shift(-h).loc[dt]/P.loc[dt]-1],axis=1).dropna()
  if len(q)>=8:a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
 a=np.array(a);print(h,len(a),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0))
valid=f.loc[:cut].notna().sum(axis=1);print('coverage',valid.mean()/15,'avg',valid.mean(),'turn',f.loc[:cut].rank(pct=True).diff().abs().mean(axis=1).mean())
out=[]
for dt in P.loc[:cut].index:
 q=pd.concat([f.loc[dt],P.shift(-10).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(q)>=8:out.append((dt,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
d=pd.DataFrame(out,columns=['date','ic']).set_index('date');print(d.groupby(d.index.year).ic.mean().to_string())
