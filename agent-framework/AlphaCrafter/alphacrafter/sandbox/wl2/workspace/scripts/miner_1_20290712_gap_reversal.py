import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in u:
 d=get_stock_daily_data(s,1800)
 if d is None or len(d)==0:d=get_index_daily_data(s,1800)
 if d is not None and len(d):F[s]=d.set_index('date')
cl=pd.DataFrame({s:x.close.astype(float) for s,x in F.items()}).sort_index(); op=pd.DataFrame({s:x.open.astype(float) for s,x in F.items()}).reindex(cl.index); hi=pd.DataFrame({s:x.high.astype(float) for s,x in F.items()}).reindex(cl.index); lo=pd.DataFrame({s:x.low.astype(float) for s,x in F.items()}).reindex(cl.index)
gap=op/cl.shift(1)-1; tr=(hi-lo)/cl; f=-(gap/(tr.rolling(20,min_periods=10).mean()+1e-12)).shift(1); fw=cl.pct_change().shift(-1)
rows=[]; sig=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.iloc[:,0].std()>1e-12 and z.iloc[:,1].std()>1e-12:
  ic=z.iloc[:,0].corr(z.iloc[:,1])
  if np.isfinite(ic): rows.append((dt,ic,len(z))); sig += [{'date':dt,'symbol':s,'signal':float(v)} for s,v in f.loc[dt].dropna().items()]
a=np.array([x[1] for x in rows]); ns=np.array([x[2] for x in rows])
print('dates',len(a),'avgN',round(ns.mean(),2),'coverage',round(ns.mean()/len(u),4))
print('daily IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/a.std(ddof=1),np.mean(a>0)))
for lab,cut in [('2027+',pd.Timestamp('2027-01-01')),('2028+',pd.Timestamp('2028-01-01')),('2029+',pd.Timestamp('2029-01-01'))]:
 q=np.array([x[1] for x in rows if x[0]>=cut]); print(lab,'dates',len(q),'IC %.6f ICIR %.6f'%(q.mean(),q.mean()/q.std(ddof=1)) if len(q)>1 else 'insufficient')
pd.DataFrame(sig).to_csv('scripts/miner_1_20290712_gap_reversal_signal.csv',index=False)
print('signal_artifact scripts/miner_1_20290712_gap_reversal_signal.csv')
