import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try: d=get_stock_daily_data(s,days=4000)
 except Exception: d=None
 if d is None or len(d)<250:
  try: d=get_index_daily_data(s,days=4000)
  except Exception: d=None
 if d is None or len(d)==0:return None
 return d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
px={s:get(s) for s in U}; px={s:v for s,v in px.items() if v is not None}; v=get('VIX')
if v is None: print('NO VIX'); raise SystemExit
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change(); vm=v.reindex(p.index).ffill(); gate=vm < vm.rolling(60,min_periods=60).median()
ret5=p.pct_change(5); med=ret5.median(axis=1); vol=r.rolling(20).std(); f=(-(ret5.sub(med,axis=0)).div(vol*np.sqrt(20))).shift(1); fr=p.pct_change(10).shift(-10)
rows=[]
for dt in f.index:
 if not bool(gate.get(dt,False)):continue
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(a),'meanIC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean(),'coverage',a.n.mean()/len(U))
for label,g in a.groupby(pd.cut(a.index.year,[2019,2023,2026,2028,2030])):print(label,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1) if len(g)>1 else np.nan)
print('recent252',a.tail(252).ic.mean(),a.tail(252).ic.mean()/a.tail(252).ic.std(ddof=1)); print('turnover',f.rank(axis=1,pct=True).loc[a.index].diff().abs().mean(axis=1).mean())
