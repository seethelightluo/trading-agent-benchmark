import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].sort_index()
P=pd.DataFrame(D).loc[:'2028-11-16']; R=P.pct_change(); defensive=R[['XAU','US10Y','CN10Y']].mean(axis=1)
# relative momentum: asset 20d return minus defensive-basket 20d return, smoothed and lagged
F=P.pct_change(20).sub(defensive.rolling(20).sum(),axis=0).rolling(3).mean().shift(1)
for h in [1,5,10,20]:
 y=P.pct_change(h).shift(-h); a=[]; nn=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);nn.append(len(z))
 a=np.array(a); print(h,len(a),np.mean(nn),a.mean(),a.mean()/(a.std(ddof=1)+1e-12),np.mean(a>0))
 if h==1:print('recent250',a[-250:].mean(),a[-250:].mean()/(a[-250:].std(ddof=1)+1e-12),'coverage',np.mean(nn)/15)
