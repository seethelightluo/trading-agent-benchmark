import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={s:get_stock_daily_data(s,3000) for s in U}
rows=[]
for s,d in D.items():
 if d is None or len(d)<80: continue
 d=d.copy(); d.date=pd.to_datetime(d.date); r=d.close.pct_change(); rng=(d.high-d.low).rolling(10).mean(); body=d.close.pct_change(10); eff=body/(rng/d.close).replace(0,np.nan); consistency=(r.gt(0).rolling(10).mean()-.5)*2; sig=eff*(1+0.5*consistency)
 for i in range(40,len(d)-10):
  if pd.notna(sig.iloc[i]): rows.append((d.date.iloc[i],s,float(sig.iloc[i]),float(d.close.iloc[i+10]/d.close.iloc[i]-1)))
x=pd.DataFrame(rows,columns=['date','s','f','r']); vals=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: vals.append((dt,g.f.corr(g.r,method='spearman'),len(g)))
o=pd.DataFrame(vals,columns=['date','ic','n']).dropna(); print('dates',len(o),'avg_n',o.n.mean(),'coverage',x.groupby('date').size().mean()/15); print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean()); print('turnover',x.assign(q=x.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='q').diff().abs().mean().mean())
for y,g in o.groupby(o.date.dt.year): print(y,round(g.ic.mean(),5),len(g))
