import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is None or len(d)<150: continue
    x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
p=pd.DataFrame(px).sort_index().ffill(); r=np.log(p).diff()
ret20=p.pct_change(20).shift(1); cs=ret20.mean(axis=1); res=ret20.sub(cs,axis=0)
vol=r.rolling(20).std().shift(1); f=(-res/vol.replace(0,np.nan))
f=f.clip(f.quantile(.05,axis=1),f.quantile(.95,axis=1),axis=0)
for h in [1,5,10,20]:
    fr=p.shift(-h)/p-1; vals=[]; dates=[]; ns=[]
    for dt in f.index:
        z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
    ic=pd.Series(vals,index=dates).dropna(); print('h',h,'dates',len(ic),'avgN',np.mean(ns),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
rank=f.rank(axis=1,pct=True); print('assets',len(p.columns),'rows',len(p),'range',p.index.min(),p.index.max(),'turnover',rank.diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean())
fr=p.shift(-10)/p-1
for label,cut in [('recent365',365),('recent180',180),('ytd',None)]:
    start=f.index.max()-pd.Timedelta(days=cut) if cut else pd.Timestamp('2032-01-01'); vals=[]
    for dt in f.loc[start:].index:
        q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(q)>=8: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
    a=pd.Series(vals).dropna(); print(label,len(a),a.mean(),a.mean()/a.std(ddof=1))
