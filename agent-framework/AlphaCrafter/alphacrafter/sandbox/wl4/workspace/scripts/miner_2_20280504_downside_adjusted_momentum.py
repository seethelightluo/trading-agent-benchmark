import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); P[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index().ffill(); R=P.pct_change()
neg2=(R.where(R<0)**2).rolling(20,min_periods=10).mean(); down=np.sqrt(neg2)
factor=P.pct_change(30)/down.replace(0,np.nan)
def calc(h):
 rows=[]
 for i in range(len(P)-h):
  z=pd.concat([factor.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rows.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
r=calc(10)
print('candidate=downside_adjusted_momentum_30d')
print('dates',len(r),'avg_n',r.n.mean(),'min_n',r.n.min(),'coverage',r.n.sum()/(len(r)*15))
mean=r.ic.mean(); sd=r.ic.std(ddof=1); print('IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(r.ic>0).mean())
rank=factor.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean(axis=1).dropna().mean())
for label,x in [('early',r.head(len(r)//2)),('late',r.tail(len(r)//2)),('recent250',r.tail(250))]: print(label,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1)*np.sqrt(252))
for h in [1,5,10,20]:
 x=calc(h); print('decay',h,'IC',x.ic.mean(),'obs',len(x))
