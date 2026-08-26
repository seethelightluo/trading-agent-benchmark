import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2035-04-29')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.loc[d.index<=cut,'close'].astype(float)
# common dates, factor: 20d relative momentum to cross-asset median, volatility scaled, breadth-conditioned
common=sorted(set.intersection(*[set(x.index) for x in px.values()]))
rows={h:[] for h in [1,5,10,20]}; dates={h:[] for h in rows}; counts=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  if dt not in p.index: continue
  loc=p.index.get_loc(dt)
  if loc<25 or loc+20>=len(p): continue
  r20=p.iloc[loc]/p.iloc[loc-20]-1
  r5=p.iloc[loc]/p.iloc[loc-5]-1
  rv=p.pct_change().iloc[max(0,loc-20):loc].std()*np.sqrt(20)
  vals[s]=(r20,rv,r5)
  fw[s]={h:p.iloc[loc+h]/p.iloc[loc]-1 for h in [1,5,10,20] if loc+h<len(p)}
 if len(vals)<8: continue
 # broad macro state from available cross-section; favor relative strength in weak breadth, damp in broad rallies
 breadth=np.mean([v[2]>0 for v in vals.values()])
 raw={s:(v[0]-np.median([x[0] for x in vals.values()]))/max(v[1],1e-5) for s,v in vals.items()}
 # conditional contrarian tilt only when breadth weak: trend signal otherwise
 # smooth bounded multiplier avoids degenerate signal
 mult=1.0 if breadth>=.5 else -0.35
 f={s:raw[s]*mult for s in vals}
 for h in rows:
  a=[]; b=[]
  for s in f:
   if h in fw[s] and np.isfinite(f[s]) and np.isfinite(fw[s][h]): a.append(f[s]); b.append(fw[s][h])
  if len(a)>=8:
   ic=spearmanr(a,b).statistic
   if np.isfinite(ic): rows[h].append(ic); dates[h].append(dt)
# report mean IC, ICIR mean/std, hit, counts, regime split
print('cut',cut.date(),'common_dates',len(common))
for h in rows:
 x=np.array(rows[h]); print('H',h,'IC %.6f ICIR %.6f n %d avg_n %.2f hit %.3f'%(x.mean(),x.mean()/x.std(ddof=1),len(x),np.mean([sum(1 for s in U if s in px and dates[h][i] in px[s].index) for i in range(len(x))]),np.mean(x>0)))
 for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-04-29')]:
  z=x[(np.array(dates[h])>=pd.Timestamp(a))&(np.array(dates[h])<=pd.Timestamp(b))]
  if len(z): print(' ',a[:4]+'-'+b[:4],len(z),round(z.mean(),6),round(z.mean()/z.std(ddof=1),4) if len(z)>1 and z.std(ddof=1)>0 else None)
# artifact factor values latest dates for provenance
out=[]
for dt in common[-300:]:
 vals={}
 for s,p in px.items():
  if dt in p.index:
   loc=p.index.get_loc(dt)
   if loc>=25:
    rv=p.pct_change().iloc[loc-20:loc].std()*np.sqrt(20)
    vals[s]=((p.iloc[loc]/p.iloc[loc-20]-1),rv,(p.iloc[loc]/p.iloc[loc-5]-1))
 if len(vals)>=8:
  breadth=np.mean([v[2]>0 for v in vals.values()]); med=np.median([v[0] for v in vals.values()]); mult=1 if breadth>=.5 else -.35
  for s,v in vals.items(): out.append({'date':dt.date().isoformat(),'symbol':s,'signal':(v[0]-med)/max(v[1],1e-5)*mult})
pd.DataFrame(out).to_csv('scripts/miner_3_20350430_breadth_relative_momentum_signal.csv',index=False)
