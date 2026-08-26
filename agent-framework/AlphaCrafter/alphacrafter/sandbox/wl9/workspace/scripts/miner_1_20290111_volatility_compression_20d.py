import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index();R=P.pct_change()
# compression: negative recent volatility relative to longer baseline, intended mean reversion/risk normalization
F=-(R.rolling(10,min_periods=10).std()/R.rolling(60,min_periods=60).std()-1)
for h in [1,5,10,20]:
 y=P.shift(-h)/P-1;v=[];ds=[];nn=[]
 for dt in F.index:
  a=pd.concat([F.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(a)>=8:v.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic);ds.append(dt);nn.append(len(a))
 z=pd.Series(v,index=pd.to_datetime(ds)).dropna()
 def q(x):return round(x.mean(),6),round(x.mean()/x.std(ddof=1),6),round((x>0).mean(),4),len(x)
 print(h,q(z),q(z.tail(252)),q(z[z.index>='2026-07-16']))
print('coverage',F.notna().mean().mean(),'dates',len(F),'mean_n',np.mean(nn),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
