import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-06-10'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]; dates=[]; ns=[]; out=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc<65 or loc+20>=len(p): continue
  r=p.pct_change(); vals[s]=(p.iloc[loc]/p.iloc[loc-20]-1,r.iloc[loc-60:loc].std()*np.sqrt(60)); fw[s]=p.iloc[loc+20]/p.iloc[loc]-1
 if len(vals)<8: continue
 med=np.median([v[0] for v in vals.values()]); f={s:-(v[0]-med)/max(v[1],1e-6) for s,v in vals.items()}
 a=[f[s] for s in f]; b=[fw[s] for s in f]; ic=spearmanr(a,b).statistic
 if np.isfinite(ic): rows.append(ic); dates.append(dt); ns.append(len(a))
 for s,v in vals.items(): out.append({'date':dt.date().isoformat(),'symbol':s,'signal':f[s]})
x=np.array(rows); print('factor=volscaled_20d_residual_reversal H20 cut',cut.date(),'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-06-10')]:
 z=x[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]; print(a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
pd.DataFrame(out).to_csv('scripts/miner_1_20350611_slow_residual_reversal_signal.csv',index=False)
