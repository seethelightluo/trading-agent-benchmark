import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; U=[s for s in U if s not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<150:d=get_index_daily_data(s,4000)
 if d is not None:D[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff(); m=r.mean(1)
# Observation-only VIX, lagged; activate residual reversal only in elevated volatility.
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v.date); vx=v.set_index('date').close.astype(float).reindex(p.index).ffill()
rv=vx.rolling(20,min_periods=15).mean(); active=(rv>rv.rolling(252,min_periods=100).median()).shift(1)
cov=r.ewm(alpha=.06,min_periods=40).cov(m); vm=m.ewm(alpha=.06,min_periods=40).var(); beta=cov.div(vm,axis=0); res=r.sub(beta.mul(m,axis=0),axis=0)
for look in [3,5,10]:
 f=(-res.rolling(look,min_periods=look).sum()/r.rolling(20,min_periods=15).std()).shift(1).where(active,0)
 y=np.log(p).shift(-1)-np.log(p); a=[];ns=[];dt=[]
 for t in f.index:
  z=pd.concat([f.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q):a.append(q);ns.append(len(z));dt.append(t)
 a=np.array(a);print('look',look,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'active',round(active.mean(),4))
 print('recent IC',round(a[np.array([x.year>=2026 for x in dt])].mean(),6))
 print('turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean(),6))
