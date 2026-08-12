import os,json,numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z=[]
for s in watch:
 d=get_stock_daily_data(s,days=5000)
 if d is None:continue
 d=d.sort_values('date'); r=pd.to_numeric(d['pct_change'],errors='coerce')/100
 # lagged inverse realized vol, cross-sectionally rank-equivalent
 f=1/(r.shift(1).rolling(30,min_periods=25).std()+1e-8)
 z.append(pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'factor':f.values,'fwd10':d.close.shift(-10).values/d.close.values-1}))
x=pd.concat(z).dropna(); art='scripts/miner_1_20290920_inverse_vol30_signal.csv';x[['date','symbol','factor']].to_csv(art,index=False)
a=[];n=[]
for d,g in x.groupby('date'):
 if len(g)>=8 and g.factor.nunique()>1:a.append(g.factor.corr(g.fwd10,method='spearman'));n.append(len(g))
a=pd.Series(a).dropna();print(json.dumps({'artifact':art,'valid_dates':len(a),'avg_instruments':np.mean(n),'IC':a.mean(),'ICIR':a.mean()/a.std(ddof=1),'hit_ratio':(a>0).mean(),'coverage':x.factor.notna().mean(),'turnover':x.sort_values(['symbol','date']).groupby('symbol').factor.apply(lambda q:(q.pct_change().abs()>.1).mean()).mean()},indent=2))
for lab,mask in [('2020-25',x.date.dt.year<=2025),('2026+',x.date.dt.year>=2026),('2028+',x.date.dt.year>=2028),('2029YTD',x.date.dt.year==2029)]:
 q=[]
 for d,g in x[mask].groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1:q.append(g.factor.corr(g.fwd10,method='spearman'))
 q=pd.Series(q).dropna();print(lab,len(q),q.mean(),q.mean()/q.std(ddof=1))
