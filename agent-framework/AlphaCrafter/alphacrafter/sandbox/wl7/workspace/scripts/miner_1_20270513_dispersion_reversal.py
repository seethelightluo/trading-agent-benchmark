import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv';f=f if os.path.exists(f) else '../persistent/index_data/'+a+'.csv'
 D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.concat(D,axis=1,sort=True).ffill().loc[:'2027-05-13']; r=p.pct_change(); vol=r.rolling(20).std()
# Cross-sectional rank of 3d reversal, conditioned on elevated breadth dispersion.
r3=p.pct_change(3); disp=r.std(axis=1).rolling(20).mean(); med=disp.rolling(120).median(); active=(disp>med).astype(float)
f=((-r3/vol).mul(active,axis=0)).shift(1); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=q.ic.mean();ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10]:
 yy=p.pct_change(h).shift(-h);z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=q.loc[lo:hi].ic;print('regime',lo,len(z),z.mean(),z.mean()/z.std(ddof=1))
q.to_csv('scripts/miner_1_20270513_dispersion_reversal_signal.csv')
