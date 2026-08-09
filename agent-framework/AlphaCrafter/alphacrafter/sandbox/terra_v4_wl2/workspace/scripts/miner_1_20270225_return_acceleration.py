import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames=[]
for s in U:
 d=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try: d=fn(s,days=1900)
  except Exception: d=None
  if d is not None and len(d): break
 if d is not None and len(d):
  x=d[['date','close']].copy(); x['symbol']=s; frames.append(x)
px=pd.concat(frames).pivot(index='date',columns='symbol',values='close').reindex(columns=U).sort_index()
f=px.pct_change(20)-px.pct_change(5); all_ic={h:[] for h in [1,3,5,10]}; dates={h:[] for h in all_ic}; ns={h:[] for h in all_ic}
for dt in f.index:
 for h in all_ic:
  z=pd.concat([f.loc[dt],px.shift(-h).loc[dt]/px.loc[dt]-1],axis=1).dropna()
  if len(z)>=8: all_ic[h].append(z.iloc[:,0].rank().corr(z.iloc[:,1],method='spearman')); dates[h].append(dt); ns[h].append(len(z))
for h in all_ic:
 v=pd.Series(all_ic[h]); print('horizon',h,'dates',len(v),'avg_n',np.mean(ns[h]),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean())
ser=pd.Series(all_ic[3],index=dates[3])
for name,sl in [('2020-2022',ser.loc['2020':'2022']),('2023-2024',ser.loc['2023':'2024']),('2025-2026',ser.loc['2025':'2026']),('2026-07 onward',ser.loc['2026-07-16':'2027-02-25'])]: print(name,len(sl),sl.mean(),sl.mean()/sl.std(ddof=1) if len(sl)>1 else np.nan)
rank=f.rank(axis=1,pct=True); tr=[]
for a,b in zip(rank.index[:-1],rank.index[1:]):
 z=pd.concat([rank.loc[a],rank.loc[b]],axis=1).dropna()
 if len(z)>=8: tr.append(np.mean(np.abs(z.iloc[:,1]-z.iloc[:,0])))
print('symbols',len(frames),'dates',len(px),'coverage',f.notna().sum().sum()/f.size,'turnover_proxy',np.mean(tr))
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('../persistent/factor_signals_miner_1_20270225_return_acceleration.csv',index=False)
