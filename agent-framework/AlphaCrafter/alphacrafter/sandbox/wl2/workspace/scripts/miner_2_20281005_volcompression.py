import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
U=[x for x in U if x not in {'DXY','USDCNY','USDJPY','EURUSD','VIX'}]
D={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)<100: d=get_index_daily_data(s,4000)
 if d is not None: D[s]=d.set_index(pd.to_datetime(d.date))['close'].astype(float)
P=pd.DataFrame(D).sort_index(); r=P.pct_change(); rv20=r.rolling(20).std(); rv60=r.rolling(60).std()
f=(P.pct_change(20)/(rv20*np.sqrt(20))).mul((rv60/rv20).clip(0.5,2.0)).shift(1)
print('universe',len(P.columns),'dates',len(P),'range',P.index.min(),P.index.max())
for h in [1,3,5,10,20]:
 ic=[]; ns=[]; turns=[]
 for i in range(len(P)-h):
  z=pd.concat([f.iloc[i],P.pct_change(h).iloc[i+h]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
  if i>0: turns.append((f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
 q=pd.Series(ic).dropna(); print('h',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),3),'turn',round(np.nanmean(turns),4),'coverage',round(np.mean(ns)/len(U),3))
print('signal_rows',int(f.notna().sum().sum()))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281005_volcompression_signal.csv',index=False)
