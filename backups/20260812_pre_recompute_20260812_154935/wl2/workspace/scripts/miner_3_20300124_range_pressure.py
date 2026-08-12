import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s,1500)
    except Exception: x=None
    if x is None or len(x)<100:
        try: x=get_index_daily_data(s,1500)
        except Exception: x=None
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')
common=sorted(set.intersection(*[set(x.index) for x in D.values()]))
close=pd.DataFrame({s:D[s].reindex(common)['close'] for s in D}); high=pd.DataFrame({s:D[s].reindex(common)['high'] for s in D}); low=pd.DataFrame({s:D[s].reindex(common)['low'] for s in D})
r=close.pct_change(); rng=(high-low).replace(0,np.nan)
clv=(close-low)/rng-0.5
f=(clv.rolling(15,min_periods=10).mean()/r.rolling(20,min_periods=15).std()).replace([np.inf,-np.inf],np.nan).rank(axis=1,pct=True)
for h in [1,5,10]:
 ic=[]; ns=[]; dates=[]; prev=None; turns=[]
 fr=f.shift(1); fut=close.pct_change(h).shift(-h)
 for dt in common:
  a=pd.concat([fr.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(a)>=8: ic.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman')); ns.append(len(a)); dates.append(dt)
  q=fr.loc[dt].rank(pct=True)
  if prev is not None: turns.append(np.mean(np.abs(q-prev)))
  prev=q
 ic=np.array(ic); ic=ic[np.isfinite(ic)]
 print({'horizon':h,'dates':len(ic),'avg_n':round(float(np.mean(ns)),2),'coverage':round(float(np.mean(ns)/15),4),'IC':round(float(ic.mean()),6),'ICIR':round(float(ic.mean()/ic.std()),6),'hit':round(float(np.mean(ic>0)),4),'turnover':round(float(np.nanmean(turns)),4)})
 for st in ['2020-01-01','2023-01-01','2026-01-01','2028-01-01','2029-01-01']:
  z=ic[[d>=pd.Timestamp(st) for d in dates]]
  if len(z)>20: print(st,round(float(z.mean()),5),round(float(z.mean()/z.std()),4),len(z))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20300124_range_pressure_signal.csv',index=False)
print('artifact',len(out),'assets',len(D),'range',common[0],common[-1])
