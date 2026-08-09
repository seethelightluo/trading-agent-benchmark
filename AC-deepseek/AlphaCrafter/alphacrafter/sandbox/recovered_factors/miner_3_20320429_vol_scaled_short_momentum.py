import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(f); d['date']=pd.to_datetime(d.date); px[a]=d.set_index('date').close.sort_index()
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# single idea: short-horizon momentum normalized by medium realized risk
f=R.rolling(5).sum()/R.rolling(20).std()
rows=[]
for i in range(20,len(P)-1):
 x=f.iloc[i]; y=R.iloc[i+1]
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((P.index[i],len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('factor=5d_return/20d_vol; dates',len(df),'meanN',df.n.mean(),'coverage_cells',df.n.sum()/(len(df)*15))
for h in [1,5,10,20]:
 vals=[]
 for i in range(20,len(P)-h):
  x=f.iloc[i]; y=P.pct_change(h).iloc[i+h]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('h',h,'n',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
print('turnover10',np.nanmean((f.rank(axis=1,pct=True)-f.shift(10).rank(axis=1,pct=True)).abs().mean(axis=1)))
for lo,hi in [('2020','2024'),('2024','2028'),('2028','2031'),('2031','2033')]:
 q=df.loc[lo:hi].ic.dropna(); print(lo,hi,'n',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('recent120',df.ic.tail(120).mean(),df.ic.tail(120).mean()/df.ic.tail(120).std(ddof=1))
