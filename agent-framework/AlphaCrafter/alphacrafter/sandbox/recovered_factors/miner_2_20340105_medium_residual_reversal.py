import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];d={}
for a in A:
 q='../persistent/stock_data/'+a+'.csv';q=q if os.path.exists(q) else '../persistent/index_data/'+a+'.csv';x=pd.read_csv(q);x.date=pd.to_datetime(x.date);d[a]=x.set_index('date').close.astype(float)
pd0=pd.DataFrame(d).sort_index();r=pd0.pct_change();res=r.sub(r.mean(axis=1),axis=0)
# Medium-horizon cross-sectional residual reversal, volatility scaled and lagged.
f=(-res.rolling(20).sum()/(r.rolling(20).std()*np.sqrt(20)+1e-8)).shift(1)
print('cells',int(f.notna().sum().sum()),'coverage',f.notna().sum().sum()/f.size,'dates',len(f),'assets',len(A))
for h in [1,5,10,20]:
 y=pd0.shift(-h)/pd0-1;ss=[];ix=[];nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ix.append(dt);nn.append(len(z))
 s=pd.Series(ss,index=ix);print(h,'IC %.6f ICIR %.6f hit %.4f dates %d N %.2f'%(s.mean(),s.mean()/s.std(),(s>0).mean(),len(s),np.mean(nn)))
 for lo,hi in [('2024','2027'),('2028','2030'),('2031','2033')]:
  q=s[(s.index>=lo)&(s.index<=hi+'-12-31')];print(lo,'%.6f %.6f %.4f %d'%(q.mean(),q.mean()/q.std(),(q>0).mean(),len(q)))
print('turn10',f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean())
