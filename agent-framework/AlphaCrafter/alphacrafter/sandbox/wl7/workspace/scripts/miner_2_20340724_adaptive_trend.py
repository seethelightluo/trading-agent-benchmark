import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is None: continue
 d=d[['date','close']].drop_duplicates('date').set_index('date').sort_index(); r=d.close.pct_change()
 # adaptive trend: 20d momentum when 60d trend positive, short reversal when negative; all lagged
 mom=d.close.shift(1)/d.close.shift(21)-1; trend=d.close.shift(1)/d.close.shift(61)-1; vol=r.rolling(60).std().shift(1)
 d['f']=np.where(trend>0,mom,-mom)/(vol*np.sqrt(20)); d.f=d.f.replace([np.inf,-np.inf],np.nan);F[s]=d
D=sorted(set().union(*[set(x.index) for x in F.values()])); out=[]
for h in [1,5,10,20]:
 ic=[]
 for dt in D:
  a=[];y=[]
  for s,d in F.items():
   if dt not in d.index:continue
   i=d.index.get_loc(dt)
   if i+h>=len(d):continue
   if np.isfinite(d.f.iloc[i]):a.append(d.f.iloc[i]);y.append(d.close.iloc[i+h]/d.close.iloc[i]-1)
  if len(a)>=8:ic.append(pd.Series(a).rank().corr(pd.Series(y).rank()))
 q=pd.Series(ic);print('h',h,'IC %.6f ICIR %.6f hit %.4f dates %d'%(q.mean(),q.mean()/q.std(),(q>0).mean(),len(q)))
# selected artifact
rows=[]
for dt in D:
 for s,d in F.items():
  if dt in d.index and np.isfinite(d.loc[dt,'f']): rows.append({'date':dt,'symbol':s,'signal':d.loc[dt,'f']})
pd.DataFrame(rows).to_csv('scripts/miner_2_20340724_adaptive_trend_signal.csv',index=False)
