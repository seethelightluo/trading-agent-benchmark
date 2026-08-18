import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index(); ret=px.pct_change()
# one idea: volatility-managed carry: inverse 20d realized vol, cross-sectionally neutralized by 5d momentum (small blend)
vol=ret.rolling(20,min_periods=15).std(); f=1/vol
for h in [1,5,10]:
 y=px.shift(-h)/px-1; a=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.array(a); print(h,len(a),round(np.mean(ns),2),a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0),np.mean(ns)/15)
print('turn', (f.rank(pct=True,axis=1).diff().abs().mean(axis=1)>.08).mean())
print('latest',f.iloc[-1].to_dict())
