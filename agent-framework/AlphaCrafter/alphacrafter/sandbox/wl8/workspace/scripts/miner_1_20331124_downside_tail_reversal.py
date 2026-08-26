import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    try: d=get_index_daily_data(s,days=5000)
    except Exception: d=None
    if d is None or len(d)<100:
      try:d=get_stock_daily_data(s,days=5000)
      except Exception:d=None
    return d
px={}
for s in U:
    d=get(s)
    if d is not None and len(d): px[s]=d.set_index('date')['close']
P=pd.DataFrame(px).sort_index(); r=np.log(P).diff()
loss=(-r.clip(upper=0)).rolling(30,min_periods=15).std(); r5=np.log(P/P.shift(5))
raw=(-r5.clip(upper=0))/(loss.rolling(5,min_periods=3).mean()*np.sqrt(5)); f=raw.rolling(3,min_periods=2).mean(); fr=np.log(P.shift(-10)/P)
rows=[]
for dt in f.index:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        if np.isfinite(ic): rows.append((dt,len(z),ic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20331124_downside_tail_reversal_signal.csv',index=False); out.to_csv('scripts/miner_1_20331124_downside_tail_reversal_ic.csv')
for name,ser in [('full',out.ic),('365d',out.ic.tail(365)),('750d',out.ic.tail(750)),('1260d',out.ic.tail(1260))]: print(name,'dates',len(ser),'IC',ser.mean(),'ICIR',ser.mean()/ser.std(ddof=1),'hit',(ser>0).mean())
print('assets',len(P.columns),'avgN',out.n.mean(),'coverage',len(out)*out.n.mean()/(len(P.index)*len(U)),'turnover_proxy',f.rank(axis=1,pct=True).diff().abs().stack().mean())
for h in [1,5,10,20]:
 yy=np.log(P.shift(-h)/P); q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(q))
