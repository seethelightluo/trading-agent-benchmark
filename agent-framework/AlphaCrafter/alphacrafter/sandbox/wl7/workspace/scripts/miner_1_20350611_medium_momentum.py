import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-06-10'); px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()['close'] for s in U}; common=sorted(set.intersection(*[set(p.index) for p in px.values()])); rows=[]; ds=[]; ns=[]; out=[]
for dt in common:
 v={}; fw={}
 for s,p in px.items():
  i=p.index.get_loc(dt)
  if i<125 or i+20>=len(p): continue
  r=p.pct_change(); v[s]=(p.iloc[i]/p.iloc[i-60]-1,r.iloc[i-60:i].std()*np.sqrt(60)); fw[s]=p.iloc[i+20]/p.iloc[i]-1
 if len(v)<8: continue
 med=np.median([z[0] for z in v.values()]); f={s:(z[0]-med)/max(z[1],1e-6) for s,z in v.items()}; ic=spearmanr(list(f.values()),[fw[s] for s in f]).statistic
 if np.isfinite(ic): rows.append(ic);ds.append(dt);ns.append(len(f))
 for s,z in v.items(): out.append({'date':dt.date().isoformat(),'symbol':s,'signal':f[s]})
x=np.array(rows);print('factor=volscaled_60d_relative_momentum H20 cut',cut.date(),'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-06-10')]:
 z=x[(np.array(ds)>=pd.Timestamp(a))&(np.array(ds)<=pd.Timestamp(b))];print(a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6))
pd.DataFrame(out).to_csv('scripts/miner_1_20350611_medium_momentum_signal.csv',index=False)
