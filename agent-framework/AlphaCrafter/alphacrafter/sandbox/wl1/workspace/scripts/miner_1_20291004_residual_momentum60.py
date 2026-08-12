import os,json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in watch:
 d=get_stock_daily_data(s,days=5000)
 if d is not None:
  d=d.sort_values('date'); rows.append(pd.DataFrame({'date':pd.to_datetime(d['date']),'symbol':s,'r':pd.to_numeric(d['pct_change'],errors='coerce').values/100}))
x=pd.concat(rows).pivot(index='date',columns='symbol',values='r').sort_index()
cum=x.shift(1).rolling(60,min_periods=45).sum(); med=cum.median(axis=1); residual=cum.sub(med,axis=0)
vol=x.shift(1).rolling(60,min_periods=45).std()*np.sqrt(60); f=residual/(vol+1e-8)
fwd=x.shift(-1).rolling(10,min_periods=10).sum().shift(-9)
y=pd.concat([pd.DataFrame({'date':f.index,'symbol':s,'factor':f[s].values,'fwd10':fwd[s].values}) for s in f.columns]).dropna().reset_index(drop=True)
art='scripts/miner_1_20291004_residual_momentum60_signal.csv'; y[['date','symbol','factor']].to_csv(art,index=False)
def calc(z):
 a=[]; ns=[]
 for d,g in z.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:a.append(g.factor.corr(g.fwd10,method='spearman'));ns.append(len(g))
 a=pd.Series(a).dropna(); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float((a>0).mean())
print(json.dumps({'artifact':art,'overall':calc(y),'coverage':float(y.factor.notna().mean()),'dates':int(y.date.nunique()),'instruments':int(y.symbol.nunique())},indent=2))
for label,mask in [('2020-25',y.date.dt.year<=2025),('2026+',y.date.dt.year>=2026),('2028+',y.date.dt.year>=2028),('2029YTD',y.date.dt.year==2029)]: print(label,calc(y[mask]))
z=y.sort_values(['symbol','date']); print('turnover',float(z.groupby('symbol').factor.apply(lambda q:(q.diff().abs()>0.15).mean()).mean()))
