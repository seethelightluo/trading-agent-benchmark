import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def data(s):
 d=get_stock_daily_data(s,4200)
 if d is None or len(d)<150:d=get_index_daily_data(s,4200)
 return d
R=[]
for s in U:
 d=data(s)
 if d is None:continue
 x=d[['date','close']].drop_duplicates('date').set_index('date').sort_index()
 ret=x.close.pct_change(); vol=ret.rolling(20).std().shift(5)*np.sqrt(20)
 # acceleration: medium trend minus short trend, both unavailable until prior day; risk normalize
 f=((x.close.pct_change(60).shift(5)-x.close.pct_change(20).shift(5))/vol).rename('f')
 y=pd.concat([f,x.close.pct_change(5).shift(-5).rename('fr')],axis=1).dropna().reset_index();y['asset']=s;R.append(y)
z=pd.concat(R,ignore_index=True); ii=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:ii.append(g.f.corr(g.fr,method='spearman'))
ii=pd.Series(ii).dropna();print('dates',len(ii),'rows',len(z),'avg_n',z.groupby('date').size().mean(),'coverage',len(z)/(z.date.nunique()*15));print('IC',ii.mean(),'ICIR',ii.mean()/ii.std(ddof=1),'hit',(ii>0).mean())
for n in [120,252,504]:
 q=ii.tail(n);print('recent',n,q.mean(),q.mean()/q.std(ddof=1))
p=z.pivot(index='date',columns='asset',values='f').rank(axis=1,pct=True);print('turnover',p.diff().abs().mean(axis=1).dropna().mean())
