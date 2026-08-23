import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data={}
for s in U:
    d=get_stock_daily_data(s, days=2600)
    if d is not None and len(d)>120:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date').sort_index()
        data[s]=d['close'].astype(float)
p=pd.DataFrame(data).sort_index().ffill(); r=p.pct_change()
mom20=p.pct_change(20); mom5=p.pct_change(5); mom60=p.pct_change(60)
vol20=r.rolling(20).std()*np.sqrt(252)
agree=(np.sign(mom5)+np.sign(mom20)+np.sign(mom60))/3
f=(mom20/vol20)*agree; f=f.sub(f.mean(axis=1),axis=0)
def calc(h):
    fr=p.shift(-h).div(p).sub(1); vals=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8:
            c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
            if np.isfinite(c): vals.append(c); ns.append(len(z))
    a=np.array(vals); turn=(f.diff().abs().sum(axis=1)/(f.abs().sum(axis=1)*2)).replace([np.inf,-np.inf],np.nan).mean()
    return len(a),a.mean(),a.mean()/(a.std(ddof=1)/np.sqrt(len(a))),np.mean(a>0),turn,np.mean(ns)
print('universe',len(data),'dates',len(p),'range',p.index.min(),p.index.max())
for h in [1,5,10,20]: print('H',h,'n avgIC ICIR hit turnover avgN',calc(h))
h=10; fr=p.shift(-h).div(p).sub(1); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if np.isfinite(c): vals.append((dt,c))
q=pd.DataFrame(vals,columns=['date','ic']); q['year']=q.date.dt.year
print(q.groupby('year').ic.agg(['count','mean']).tail(12).to_string())
print('coverage',f.notna().sum(axis=1).mean()/len(U))
