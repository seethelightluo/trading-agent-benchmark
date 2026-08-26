import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-05-13'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]; dates=[]; ns=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc<25 or loc+10>=len(p): continue
  vals[s]=((p.iloc[loc]/p.iloc[loc-20]-1),p.pct_change().iloc[loc-20:loc].std()*np.sqrt(20),(p.iloc[loc]/p.iloc[loc-5]-1))
  fw[s]={h:p.iloc[loc+h]/p.iloc[loc]-1 for h in [1,5,10]}
 if len(vals)<8: continue
 breadth=np.mean([v[2]>0 for v in vals.values()]); med=np.median([v[0] for v in vals.values()]); mult=1 if breadth>=.5 else -.35
 f={s:(v[0]-med)/max(v[1],1e-5)*mult for s,v in vals.items()}
 a=[];b=[]
 for s in f:
  if np.isfinite(f[s]) and np.isfinite(fw[s][1]):a.append(f[s]);b.append(fw[s][1])
 ic=spearmanr(a,b).statistic
 if np.isfinite(ic):rows.append(ic);dates.append(dt);ns.append(len(a))
x=np.array(rows); print('factor=conditioned20d_relative_momentum cut',cut.date(),'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-05-13')]:
 z=x[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]; print(a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
out=[]
for dt in common:
 vals={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc>=25:
   vals[s]=((p.iloc[loc]/p.iloc[loc-20]-1),p.pct_change().iloc[loc-20:loc].std()*np.sqrt(20),(p.iloc[loc]/p.iloc[loc-5]-1))
 if len(vals)>=8:
  med=np.median([v[0] for v in vals.values()]); mult=1 if np.mean([v[2]>0 for v in vals.values()])>=.5 else -.35
  for s,v in vals.items():out.append({'date':dt.date().isoformat(),'symbol':s,'signal':(v[0]-med)/max(v[1],1e-5)*mult})
pd.DataFrame(out).to_csv('scripts/miner_3_20350513_clv_pressure_signal.csv',index=False)
