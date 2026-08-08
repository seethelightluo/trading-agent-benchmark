import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; d={}
for a in A:
 p='../persistent/stock_data/'+a+'.csv'; p=p if os.path.exists(p) else '../persistent/index_data/'+a+'.csv'
 x=pd.read_csv(p); x.date=pd.to_datetime(x.date); d[a]=x.set_index('date').close.astype(float)
p=pd.DataFrame(d).sort_index(); r=p.pct_change();
# Short-term reversal damped by volatility: negative 5-day return divided by vol20, lagged
f=(-r.rolling(5).sum()/(r.rolling(20).std()*np.sqrt(20)+1e-8)).shift(1)
print('cells',f.notna().sum().sum(),'coverage',f.notna().sum().sum()/f.size)
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ix=[]; nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ix.append(dt);nn.append(len(z))
 s=pd.Series(ss,index=ix); print(h,'IC %.6f ICIR %.6f hit %.4f dates %d N %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(nn)))
 for lo,hi in [('2024','2027'),('2028','2030'),('2031','2033')]:
  q=s[(s.index>=lo)&(s.index<=hi+'-12-31')];print(lo, '%.6f %.6f %d'%(q.mean(),q.mean()/q.std(),len(q)))
print('turn10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
