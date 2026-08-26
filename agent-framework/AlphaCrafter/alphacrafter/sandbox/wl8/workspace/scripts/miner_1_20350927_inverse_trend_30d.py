import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=get_stock_daily_data(s, days=10000)
    if d is None or len(d)<100: d=get_index_daily_data(s, days=10000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); px[s]=x.set_index('date').close
panel=pd.DataFrame(px).sort_index(); fac=-(panel.shift(1)/panel.shift(31)-1); fwd=panel.shift(-10)/panel-1
rows=[]; sigrows=[]
for dt in panel.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
  for sym,v in fac.loc[dt].dropna().items(): sigrows.append((dt,sym,float(v)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); mean=r.ic.mean(); sd=r.ic.std(ddof=1); icir=mean/sd*np.sqrt(len(r))
turn=[]; prev=None
for dt,g in pd.DataFrame(sigrows,columns=['date','symbol','signal']).groupby('date'):
 q=g.set_index('symbol').signal.rank(pct=True)
 if prev is not None:
  c=q.index.intersection(prev.index)
  if len(c)>=8: turn.append(1-spearmanr(q[c],prev[c]).statistic)
 prev=q
print('dates',len(r),'avg_n',r.n.mean(),'coverage',len(sigrows)/(len(panel)*len(U)))
print('ic',mean,'icir',icir,'hit',(r.ic>0).mean(),'turnover',np.nanmean(turn))
for h in [1,5,20]:
 rr=[]; fw=panel.shift(-h)/panel-1
 for dt in panel.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(rr),len(rr))
for n in [365,750,1260]: print('recent',n,r.ic.tail(n).mean(),min(n,len(r)))
r.to_csv('scripts/miner_1_20350927_inverse_trend_30d_ic.csv'); pd.DataFrame(sigrows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20350927_inverse_trend_30d_signal.csv',index=False)
