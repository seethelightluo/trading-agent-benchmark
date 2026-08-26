import numpy as np, pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-10-03'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
# Range-expansion continuation: recent 5d directional move, scaled by volatility,
# and weighted toward assets whose current 5d range is expanding versus its 60d baseline.
rows=[]
for s,x in D.items():
 c=x.close.astype(float); tr=(x.high-x.low)/c.replace(0,np.nan)
 r5=c.pct_change(5); vol=tr.rolling(20,min_periods=15).median()
 expansion=(tr.rolling(5,min_periods=4).mean()/tr.rolling(60,min_periods=40).median()).clip(0.5,2.0)
 f=(r5/vol)*np.sqrt(expansion)
 for h in [1,5,10,20]:
  rows.append(pd.DataFrame({'date':c.index,'symbol':s,'factor':f,'fwd':c.shift(-h)/c-1,'h':h}))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
for h,g0 in a.groupby('h'):
 out=[]
 for d,g in g0.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>=3 and g.fwd.nunique()>=3: out.append((d,g.factor.corr(g.fwd),len(g)))
 ic=pd.DataFrame(out,columns=['date','ic','n']); vals=ic.ic.dropna()
 print('H',h,'dates',len(ic),'avgN',ic.n.mean(),'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1),'hit',(vals>0).mean(),'thirds',[vals.iloc[i*len(vals)//3:(i+1)*len(vals)//3].mean() for i in range(3)])
 if h==10: a[a.h==h][['date','symbol','factor']].to_csv('scripts/miner_3_20321004_range_expansion_signal.csv',index=False)
print('instruments',len(D),'cutoff',cut.date())
