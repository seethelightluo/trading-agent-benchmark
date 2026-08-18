import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
 p=os.path.join('../persistent/stock_data',s+'.csv')
 if not os.path.exists(p): p=os.path.join('../persistent/index_data',s+'.csv')
 d=pd.read_csv(p); d.columns=[str(x).lower() for x in d.columns]
 dt=[x for x in d.columns if x in ('date','datetime','trade_date')][0]; c=[x for x in d.columns if x in ('close','收盘')][0]
 return pd.DataFrame({'date':pd.to_datetime(d[dt]),'close':pd.to_numeric(d[c],errors='coerce')}).dropna().drop_duplicates('date').set_index('date').sort_index()
P={s:load(s) for s in U}
all_dates=sorted(set().union(*[set(x.index) for x in P.values()]))
rows=[]
for s,d in P.items():
 c=d.close
 # lagged at t: all inputs through t-1; risk-adjusted 20d return, downside vol
 r=np.log(c/c.shift(20)); dr=np.log(c/c.shift(1)); down=dr.where(dr<0,0).rolling(20).std()
 f=(r/down.replace(0,np.nan)).shift(1)
 fr=np.log(c.shift(-10)/c)
 z=pd.concat([f.rename('f'),fr.rename('y')],axis=1); z['s']=s; rows.append(z)
x=pd.concat(rows).reset_index().rename(columns={'index':'date'}); x=x[x.date<=pd.Timestamp('2030-02-06')]
ics=[]; ns=[]
for dt,g in x.groupby('date'):
 g=g.dropna()
 if len(g)>=8:
  ics.append(spearmanr(g.f,g.y).statistic); ns.append(len(g))
a=np.array(ics); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/np.std(a,ddof=1),'hit',np.mean(a>0),'recent250',np.mean(a[-250:])/np.std(a[-250:],ddof=1),'coverage',len(x.dropna())/len(x),'turnover_proxy',np.nan)
for h in [1,5,10,20]:
 rr=[]
 for s,d in P.items():
  c=d.close; f=(np.log(c/c.shift(20))/np.log(c/c.shift(1)).where(np.log(c/c.shift(1))<0,0).rolling(20).std()).shift(1); y=np.log(c.shift(-h)/c)
  rr.append(pd.concat([f.rename('f'),y.rename('y')],axis=1).assign(s=s))
 q=pd.concat(rr)
 vals=[]
 for dt,g in q.groupby(level=0):
  g=g.dropna()
  if len(g)>=8: vals.append(spearmanr(g.f,g.y).statistic)
 v=np.array(vals); print('h',h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1))
