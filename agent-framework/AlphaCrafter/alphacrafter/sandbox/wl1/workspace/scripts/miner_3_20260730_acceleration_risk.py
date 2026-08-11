import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for s in U},axis=1).sort_index(); r=p.pct_change()
# Trend acceleration normalized by recent risk: short momentum minus slow momentum, scaled by 20d vol.
acc=(p/p.shift(10)-1)-(p/p.shift(40)-1)
vol=r.rolling(20,min_periods=15).std()
f=acc/(vol+1e-8)
print('rows',len(p),'assets',len(U),'cutoff',cut.date())
for h in [5,10,20]:
 v=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:v.append(spearmanr(q.f,q.y).statistic);ns.append(len(q))
 x=np.array(v);print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
v=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(q)>=8 and q.f.nunique()>1:v.append((p.index[i],spearmanr(q.f,q.y).statistic))
z=pd.Series([a for _,a in v],index=pd.DatetimeIndex([d for d,_ in v]));print('annual10d',{int(y):round(z[z.index.year==y].mean(),6) for y in sorted(z.index.year.unique())})
turn=[]
for i in range(10,len(f)):
 a=f.iloc[i].rank(pct=True);b=f.iloc[i-10].rank(pct=True);ix=a.dropna().index.intersection(b.dropna().index)
 if len(ix)>=8:turn.append(np.abs(a[ix]-b[ix]).mean())
print('turnover10d',round(np.mean(turn),6),'turn_dates',len(turn))
