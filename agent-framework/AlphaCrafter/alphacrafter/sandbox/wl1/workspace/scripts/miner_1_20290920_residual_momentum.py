import os,json
import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr=[]
for s in watch:
 d=get_stock_daily_data(s,days=5000)
 if d is None: continue
 d=d.sort_values('date'); r=pd.to_numeric(d['pct_change'],errors='coerce')/100
 allr.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'r':r.values,'close':d.close.values}))
x=pd.concat(allr).pivot(index='date',columns='symbol',values='r').sort_index()
# lagged residual momentum: 20d asset return relative to cross-asset median, divided by idiosyncratic 40d vol
m=x.shift(1).rolling(20,min_periods=15).sum(); bench=m.median(axis=1)
res=m.sub(bench,axis=0)
vol=x.shift(1).rolling(40,min_periods=30).std()*np.sqrt(20)
f=res/(vol+1e-8)
# forward 10d returns
fwd=x.shift(-1).rolling(10,min_periods=10).sum().shift(-9)
long=[]
for s in f.columns:
 long.append(pd.DataFrame({'date':f.index,'symbol':s,'factor':f[s].values,'fwd10':fwd[s].values}))
y=pd.concat(long).dropna()
os.makedirs('scripts',exist_ok=True); art='scripts/miner_1_20290920_residual_momentum_signal.csv'; y[['date','symbol','factor']].to_csv(art,index=False)
ics=[]; ns=[]
for d,g in y.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1: ics.append(g.factor.corr(g.fwd10,method='spearman'));ns.append(len(g))
a=pd.Series(ics).dropna(); print(json.dumps({'artifact':art,'valid_dates':len(a),'avg_instruments':np.mean(ns),'IC':a.mean(),'ICIR':a.mean()/a.std(ddof=1),'hit_ratio':(a>0).mean(),'coverage':y.factor.notna().mean(),'turnover':y.sort_values(['symbol','date']).groupby('symbol').factor.apply(lambda z:(z.diff().abs()>0.15).mean()).mean()},indent=2))
for label,mask in [('2020-25',y.date.dt.year<=2025),('2026+',y.date.dt.year>=2026),('2028+',y.date.dt.year>=2028),('2029YTD',y.date.dt.year==2029)]:
 q=[]
 for d,g in y[mask].groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:q.append(g.factor.corr(g.fwd10,method='spearman'))
 q=pd.Series(q).dropna();print(label,len(q),q.mean(),q.mean()/q.std(ddof=1))
