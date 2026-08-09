import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
R={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].pct_change() for s in U}
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].pct_change().rename('DXY')
r=pd.DataFrame(R).join(d,how='inner').loc[:'2026-07-15'];
for w in [20,40,60,90,120]:
 v=r.DXY.rolling(w,min_periods=max(12,int(w*.75))).var(); f=pd.DataFrame({s:-r[s].rolling(w,min_periods=max(12,int(w*.75))).cov(r.DXY)/v for s in U})
 for h in [1,5]:
  y=r[U].shift(-1).rolling(h).sum(); a=[]; ns=[]
  for dt in f.index:
   z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
  a=np.array(a);print('window',w,'h',h,'N',len(a),'names',round(np.mean(ns),2),'IC',round(np.mean(a),5),'ICIR',round(np.mean(a)/np.std(a,ddof=1),5),'hit',round(np.mean(a>0),3))
