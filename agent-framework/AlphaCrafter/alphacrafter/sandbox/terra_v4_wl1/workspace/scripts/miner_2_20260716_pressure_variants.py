import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; R={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); d=d.set_index('date'); z=(d.high-d.low).replace(0,np.nan); R[s]=(2*(d.close-d.low)/z-1,(d.close-d.open)/z,d.close)
for name,fun in [('clv3',lambda a,b:a.rolling(3).mean()),('clv1',lambda a,b:a),('candle3',lambda a,b:b.rolling(3).mean()),('mix3',lambda a,b:(.6*a+.4*b).rolling(3).mean())]:
 rows=[]
 for s,(a,b,c) in R.items():
  f=-fun(a,b); rr=c.pct_change().shift(-1); rows.append(pd.DataFrame({'date':c.index,'f':f,'r':rr,'s':s}))
 x=pd.concat(rows,ignore_index=True).dropna(); q=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:q.append(g.f.corr(g.r,method='spearman'))
 q=pd.Series(q).dropna(); print(name,len(q),q.mean(),q.mean()/q.std(ddof=1), (q>0).mean())
