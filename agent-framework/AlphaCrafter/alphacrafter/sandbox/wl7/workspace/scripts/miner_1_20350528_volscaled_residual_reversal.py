import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-05-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rows=[]; dates=[]; ns=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc<45 or loc+10>=len(p): continue
  r=p.pct_change()
  vals[s]=((p.iloc[loc]/p.iloc[loc-5]-1), r.iloc[loc-20:loc].std()*np.sqrt(20), r.iloc[loc-5:loc].std()*np.sqrt(5))
  fw[s]=p.iloc[loc+10]/p.iloc[loc]-1
 if len(vals)<8: continue
 # residualize short return against cross-section median, then inverse volatility; activate after broad shock
 med=np.median([v[0] for v in vals.values()]); crossvol=np.std([v[0] for v in vals.values()])
 f={s:-(v[0]-med)/max(v[1],1e-6)*(1+0.5*(v[2]>np.median([z[2] for z in vals.values()]))) for s,v in vals.items()}
 a=[f[s] for s in f if np.isfinite(f[s])]; b=[fw[s] for s in f if np.isfinite(f[s])]
 ic=spearmanr(a,b).statistic
 if np.isfinite(ic): rows.append(ic); dates.append(dt); ns.append(len(a))
x=np.array(rows); print('factor=vol_scaled_5d_residual_reversal cut',cut.date(),'dates',len(x),'avgN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-05-27')]:
 z=x[(np.array(dates)>=pd.Timestamp(a))&(np.array(dates)<=pd.Timestamp(b))]; print(a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),6) if len(z)>1 else None)
out=[]
for dt in common:
 vals={}
 for s,p in px.items():
  loc=p.index.get_loc(dt)
  if loc>=45:
   r=p.pct_change(); vals[s]=((p.iloc[loc]/p.iloc[loc-5]-1),r.iloc[loc-20:loc].std()*np.sqrt(20),r.iloc[loc-5:loc].std()*np.sqrt(5))
 if len(vals)>=8:
  med=np.median([v[0] for v in vals.values()]); q=np.median([z[2] for z in vals.values()]);
  for s,v in vals.items(): out.append({'date':dt.date().isoformat(),'symbol':s,'signal':-(v[0]-med)/max(v[1],1e-6)*(1+0.5*(v[2]>q))})
pd.DataFrame(out).to_csv('scripts/miner_1_20350528_volscaled_residual_reversal_signal.csv',index=False)
