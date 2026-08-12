import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)<200: d=get_index_daily_data(s,1800)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); rows=[]; sig=[]
# A trend-agreement, risk-scaled medium momentum: 10d return, gated by 20/60d trend signs
for t in range(70,len(P)-1):
 vals={}
 for s in P:
  r=R[s]
  if t<61 or not np.isfinite(r.iloc[t-1]): continue
  m10=P[s].iloc[t]/P[s].iloc[t-10]-1
  m20=P[s].iloc[t]/P[s].iloc[t-20]-1
  m60=P[s].iloc[t]/P[s].iloc[t-60]-1
  v=r.iloc[t-19:t+1].std()
  if np.isfinite(m10+m20+m60+v) and v>1e-8:
   agree=(np.sign(m20)==np.sign(m60))
   vals[s]=(m10/v)*(1.0 if agree else -0.5)
 q=pd.concat([pd.Series(vals),R.iloc[t+1].reindex(vals)],axis=1).dropna()
 if len(q)>=8:
  rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q))); sig.append(pd.Series(vals,name=P.index[t]))
a=np.array([x[1] for x in rows]); n=np.array([x[2] for x in rows])
print('dates',len(a),'avgN',round(n.mean(),2),'coverage',round(n.mean()/len(P.columns),4),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 b=a[[x[0]>=cut for x in rows]]; print(lab,'dates',len(b),'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)))
pd.DataFrame(sig).to_csv('scripts/miner_1_20290823_agreement_momentum_signal.csv',index_label='date')
