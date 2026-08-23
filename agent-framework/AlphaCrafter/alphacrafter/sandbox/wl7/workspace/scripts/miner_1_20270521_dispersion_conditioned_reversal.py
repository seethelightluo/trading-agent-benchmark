import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+a+'.csv'
 D[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:'2027-05-21','close'].astype(float)
p=pd.concat(D,axis=1,sort=True).ffill(); r=p.pct_change(); r3=p.pct_change(3); vol=r.rolling(20).std()
disp=r.apply(lambda x:x.std(),axis=1).rolling(20).mean(); scale=1+disp/disp.rolling(120).median()
f=(-r3/vol).mul(scale,axis=0).shift(1); fw=r.shift(-1)
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
q=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('dates',len(q),'avg_n',round(q.n.mean(),2),'coverage',round(q.n.mean()/15,4))
print('IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),4),'hit',round((q.ic>0).mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for h in [1,5,10,20]:
 yy=p.pct_change(h).shift(-h); z=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(a)>=8:z.append(spearmanr(a.iloc[:,0],a.iloc[:,1]).statistic)
 print('decay',h,round(np.nanmean(z),6),len(z))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-05-21')]:
 z=q.loc[lo:hi].ic; print('regime',lo,len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),4))
q.to_csv('scripts/miner_1_20270521_dispersion_conditioned_reversal_signal.csv'); print('artifact written')
