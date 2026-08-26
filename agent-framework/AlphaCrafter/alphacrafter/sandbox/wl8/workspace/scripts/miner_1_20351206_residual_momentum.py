import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: x=get_stock_daily_data(s,days=4200)
 except Exception:
  try: x=get_index_daily_data(s,days=4200)
  except Exception: x=None
 if x is not None and len(x)>300:
  x=x.sort_values('date').drop_duplicates('date').set_index('date'); D[s]=x.close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change(); m=r.mean(axis=1)
# residual cumulative return over 60 sessions, scaled by residual volatility
res=r.sub(m,axis=0); f=res.rolling(60).sum()/(res.rolling(40).std()*np.sqrt(40)+1e-12)
fr=p.shift(-10)/p-1
rows=[]; sig=[]
for dt in f.index:
 z=pd.concat([f.loc[dt].rename('f'),fr.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,z.f.corr(z.y,method='spearman'),len(z)))
  for sym,val in f.loc[dt].items():
   if pd.notna(val): sig.append((dt,sym,val))
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('loaded',len(D),'dates',len(ic),'avg_n',round(ic.n.mean(),2),'coverage',round(f.notna().sum().sum()/f.size,4))
print('IC',round(ic.ic.mean(),6),'std',round(ic.ic.std(),6),'ICIR',round(ic.ic.mean()/ic.ic.std(),6),'hit',round((ic.ic>0).mean(),4))
for n in [365,750,1260]:
 q=ic.tail(n).ic; print('recent',n,round(q.mean(),6),round(q.mean()/q.std(),6),len(q))
for h in [1,5,20]:
 y=p.shift(-h)/p-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,round(np.nanmean(a),6),len(a))
ranks=f.rank(axis=1,pct=True); to=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8:to.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',round(np.mean(to),6))
pd.DataFrame(sig,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20351206_residual_momentum_signal.csv',index=False)
ic.reset_index().to_csv('scripts/miner_1_20351206_residual_momentum_ic.csv',index=False)
