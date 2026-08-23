import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'; f=f if os.path.exists(f) else '../persistent/index_data/'+a+'.csv'
 D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:'2027-05-21','close']
p=pd.concat(D,axis=1,sort=True).ffill(); r=p.pct_change(); r3=p.pct_change(3); vol=r.rolling(20).std()
# cross-sectional residual reversal, with asset-specific volatility scaling
f=(-(r3.sub(r3.mean(axis=1),axis=0))/vol).shift(1); y=r.shift(-1); rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=q.ic.mean();ir=ic/q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 z=[];yy=p.pct_change(h).shift(-h)
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(z))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2027')]:
 z=q.loc[lo:hi].ic;print('regime',lo,len(z),z.mean(),z.mean()/z.std(ddof=1))
q.to_csv('scripts/miner_1_20270521_residual_reversal_signal.csv')
