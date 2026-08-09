import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index(); r=p.pct_change(); m=r.mean(axis=1)
# beta-neutral residual 60d trend, removing common cross-asset market move
beta=r.rolling(60).cov(m).div(m.rolling(60).var(),axis=0)
f=p.pct_change(60).sub(beta.mul(m.rolling(60).sum(),axis=0)).div(r.rolling(60).std()*np.sqrt(60))
a=[];ns=[]
for i in range(60,len(p)-1):
 z=pd.concat([f.iloc[i],p.iloc[i+1]/p.iloc[i]-1],axis=1).dropna()
 if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(a);print('n',len(a),'avgN',np.mean(ns),'coverage',sum(ns)/(len(ns)*15),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
for h in [250,500]:
 b=a[-h:];print('recent',h,b.mean(),b.mean()/b.std(ddof=1))
