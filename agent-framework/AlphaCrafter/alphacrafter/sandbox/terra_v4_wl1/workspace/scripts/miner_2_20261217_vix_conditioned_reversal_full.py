import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'; macro='../persistent/index_data'
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}).sort_index().loc[:cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close'].loc[:cut]
v5=v.pct_change(5).reindex(P.index).ffill(); r5=P.pct_change(5); sh=v5.clip(-.5,.5)
f=(-r5.mul(1+sh,axis=0)).where(sh>0,-r5*.5)
Y=P.shift(-1).div(P)-1
rows=[]
for d in P.index:
 q=pd.concat([f.loc[d].rename('signal'),Y.loc[d].rename('forward_1d')],axis=1).dropna()
 if len(q)>=8:
  ic=q.signal.corr(q.forward_1d)
  for s,row in q.iterrows(): rows.append({'date':d,'symbol':s,'signal':row.signal,'forward_1d':row.forward_1d,'cross_section_ic':ic})
pd.DataFrame(rows).to_csv('scripts/miner_2_20261217_vix_conditioned_reversal_signal.csv',index=False)
print('rows',len(rows),'dates',pd.DataFrame(rows).date.nunique())
