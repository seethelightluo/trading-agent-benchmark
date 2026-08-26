import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; S={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.drop_duplicates('date').set_index('date').sort_index();c=d.close;r=c.pct_change(); low=c.rolling(60,min_periods=45).min(); rec=c/low-1; v=r.rolling(30,min_periods=20).std();
 # medium-term recovery slope, smoothed, downside-risk adjusted
 raw=(rec-rec.shift(20))/(v*np.sqrt(20)+1e-12); f=raw.rolling(5,min_periods=3).mean().shift(1);S[s]=pd.DataFrame({'f':f,'close':c})
rows=[]
for s,x in S.items():
 y=x.close.pct_change(10).shift(-10);z=pd.concat([x.f,y.rename('y')],axis=1).dropna();z['s']=s;rows.append(z.reset_index())
a=pd.concat(rows);v=[]
for dt,g in a.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append((dt,g.f.corr(g.y),len(g)))
i=pd.DataFrame(v,columns=['date','ic','n']).set_index('date');r=a.pivot(index='date',columns='s',values='f').rank(axis=1,pct=True);to=r.diff().abs().mean(axis=1).mean()/2
print('assets',len(S),'dates',len(i),'avgN',i.n.mean(),'coverage',len(a)/(len(set(a.date))*len(U)),'IC',i.ic.mean(),'ICIR',i.ic.mean()/i.ic.std(ddof=1),'hit',(i.ic>0).mean(),'recent365',i.tail(365).ic.mean()/i.tail(365).ic.std(ddof=1),'turnover',to)
