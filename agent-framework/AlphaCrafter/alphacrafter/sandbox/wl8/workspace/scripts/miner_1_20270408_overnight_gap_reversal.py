import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 p='../persistent/stock_data/'+a+'.csv'
 x=pd.read_csv(p)
 x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').set_index('date')
 D[a]=x
# gap from prior close to today's open, signal at today's close, next-day close return
rows=[]
for a,x in D.items():
 x=x.copy(); x['gap']=x['open']/x['close'].shift(1)-1
 x['fwd1']=x['close'].shift(-1)/x['close']-1
 x['fwd5']=x['close'].shift(-5)/x['close']-1
 x['fwd10']=x['close'].shift(-10)/x['close']-1
 for dt,r in x[['gap','fwd1','fwd5','fwd10']].dropna().iterrows(): rows.append((dt,a,r.gap,r.fwd1,r.fwd5,r.fwd10))
z=pd.DataFrame(rows,columns=['date','asset','f','fwd1','fwd5','fwd10'])
for h in ['fwd1','fwd5','fwd10']:
 vals=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8: vals.append(spearmanr(g.f,g[h]).statistic)
 vals=np.array(vals); print(h,'dates',len(vals),'mean',np.nanmean(vals),'icir',np.nanmean(vals)/np.nanstd(vals,ddof=1),'hit',np.mean(vals>0),'nobs',len(z),'coverage',len(z)/(len(assets)*len(set(z.date))))
# regimes: annual
for yr,g in z.groupby(z.date.dt.year):
 vals=[]
 for dt,q in g.groupby('date'):
  if len(q)>=8: vals.append(spearmanr(q.f,q.fwd1).statistic)
 print(yr,len(vals),round(np.nanmean(vals),4) if vals else None)
