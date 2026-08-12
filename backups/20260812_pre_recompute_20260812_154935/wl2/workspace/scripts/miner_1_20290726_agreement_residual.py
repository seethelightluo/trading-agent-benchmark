import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,1500)
 if d is None or len(d)<100: d=get_index_daily_data(s,1500)
 if d is not None: px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); R=P.pct_change(); m=R.mean(axis=1)
# Cross-asset residual reversal with a trend-agreement gate: fade residual moves only when
# the asset's 20d trend agrees with the market trend, reducing mixed-regime noise.
rows=[]
for t in range(60,len(P)-1):
 vals={}
 for s in P.columns:
  z=pd.concat([R[s].iloc[t-39:t+1],m.iloc[t-39:t+1]],axis=1).dropna()
  if len(z)<25: continue
  vx=z.iloc[:,1].var(ddof=1)
  if vx<1e-12: continue
  beta=z.iloc[:,0].cov(z.iloc[:,1])/vx; vol=z.iloc[:,0].std()
  if vol<1e-8: continue
  resid=(R[s]-beta*m).iloc[t-4:t+1].sum()
  asset_tr=P[s].iloc[t]/P[s].iloc[t-20]-1
  market_tr=P.mean(axis=1).iloc[t]/P.mean(axis=1).iloc[t-20]-1
  if np.sign(asset_tr)==np.sign(market_tr) and np.sign(asset_tr)!=0:
   vals[s]=-resid/(vol*np.sqrt(5))
 f=pd.Series(vals).dropna(); q=pd.concat([f,R.iloc[t+1].reindex(f.index)],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].std()>0 and q.iloc[:,1].std()>0:
  ic=q.iloc[:,0].corr(q.iloc[:,1])
  if np.isfinite(ic): rows.append((P.index[t],ic,len(q)))
a=np.array([x[1] for x in rows]); n=np.array([x[2] for x in rows])
print('dates',len(a),'avgN',n.mean(),'coverage',n.mean()/len(P.columns),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for label,cut in [('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 b=a[[x[0]>=cut for x in rows]]
 print(label,'dates',len(b),'IC %.6f ICIR %.6f'%(b.mean(),b.mean()/b.std(ddof=1)) if len(b)>1 else 'insufficient')
# save reproducible signal artifact for audit
out=[]
for d,ic,nn in rows: out.append({'date':d.strftime('%Y-%m-%d'),'ic':float(ic),'n':int(nn)})
pd.DataFrame(out).to_csv('scripts/miner_1_20290726_agreement_residual_signal.csv',index=False)
