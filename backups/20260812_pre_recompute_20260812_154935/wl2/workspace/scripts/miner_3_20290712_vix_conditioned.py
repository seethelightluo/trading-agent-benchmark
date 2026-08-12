import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100:d=get_index_daily_data(s,1500)
 if d is not None:px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change();
# Observation-only VIX shock: contrarian asset 3d return, activated after a 2d VIX rise.
v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.set_index('date')['close'].astype(float).reindex(P.index).ffill()
rows=[]
for t in range(30,len(P)-1):
 if pd.isna(v.iloc[t]) or v.iloc[t-2:t+1].isna().any():continue
 shock=v.iloc[t]/v.iloc[t-2]-1
 vals={}
 for s in P:
  rr=R[s]
  vol=rr.iloc[t-19:t+1].std()
  if vol>1e-8:
   # smooth activation: rising VIX gives reversal, falling VIX gives short-term trend
   x=rr.iloc[t-2:t+1].sum()
   vals[s]=(-x if shock>0 else x)/(vol*np.sqrt(20))
 f=pd.Series(vals).dropna(); z=pd.concat([f,R.iloc[t+1].reindex(f.index)],axis=1).dropna()
 if len(z)>=8: rows.append((P.index[t],z.iloc[:,0].corr(z.iloc[:,1]),len(z),f))
ics=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('VIX-conditioned 3d reversal/trend; dates',len(rows),'avgN',ns.mean(),'coverage',ns.mean()/len(P.columns))
print('IC %.6f ICIR %.6f hit %.4f'%(ics.mean(),ics.mean()/ics.std(ddof=1),np.mean(ics>0)))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 a=ics[[r[0]>=cut for r in rows]]
 print(lab,'dates',len(a),'IC %.6f ICIR %.6f'%(a.mean(),a.mean()/a.std(ddof=1)) if len(a)>2 else 'NA')
turn=[]; prev=None
for _,_,_,f in rows:
 q=f.rank(pct=True)
 if prev is not None:turn.append(np.mean(abs(q.reindex(prev.index).fillna(.5)-prev.reindex(q.index).fillna(.5))))
 prev=q
print('rank turnover %.6f'%np.mean(turn))
for h in [3,5,10]:
 a=[]
 for date,_,_,f in rows:
  j=P.index.get_loc(date)
  if j+h<len(P):
   z=pd.concat([f,(P.iloc[j+h]/P.iloc[j]-1).reindex(f.index)],axis=1).dropna()
   if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'IC %.6f ICIR %.6f dates %d'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),len(a)))
