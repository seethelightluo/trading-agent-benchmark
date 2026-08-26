import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2035-05-13'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(p.index) for p in px.values()]))
Hs=[1,5,10,20]; out={h:[] for h in Hs}; dts={h:[] for h in Hs}; sigrows=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  if dt not in p.index: continue
  i=p.index.get_loc(dt)
  if i<65 or i+20>=len(p): continue
  r10=p.iloc[i]/p.iloc[i-10]-1; r40=p.iloc[i]/p.iloc[i-40]-1
  vol=p.pct_change().iloc[i-20:i].std()*np.sqrt(20)
  vals[s]=(r10,r40,vol)
  fw[s]={h:p.iloc[i+h]/p.iloc[i]-1 for h in Hs}
 if len(vals)<8: continue
 # dispersion of 10d returns; trend continuation in orderly low-dispersion markets,
 # contrarian in shock/high-dispersion markets, smoothly scaled and cross-sectionally centered
 r=np.array([v[0] for v in vals.values()]); disp=np.std(r); med=np.median(r)
 # high dispersion switches to modest reversal; low dispersion follows medium-term trend
 w=np.clip((disp-0.045)/0.035,0,1)
 f={s:((v[0]-med)*(1-w)+(-0.45)*(v[0]-med)*w)/max(v[2],1e-5) + 0.20*((v[1]-np.median([x[1] for x in vals.values()]))/max(v[2],1e-5)) for s,v in vals.items()}
 for h in Hs:
  a=[]; b=[]
  for s in f:
   if np.isfinite(f[s]) and np.isfinite(fw[s][h]): a.append(f[s]); b.append(fw[s][h])
  if len(a)>=8:
   z=spearmanr(a,b).statistic
   if np.isfinite(z): out[h].append(z); dts[h].append(dt)
for h in Hs:
 x=np.array(out[h]); print('H%d IC %.6f ICIR %.6f n %d hit %.3f avgN %.2f'%(h,x.mean(),x.mean()/x.std(ddof=1),len(x),np.mean(x>0),len(U)))
 for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-05-13')]:
  q=x[(np.array(dts[h])>=pd.Timestamp(a))&(np.array(dts[h])<=pd.Timestamp(b))]
  if len(q)>1: print(a[:4],len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),4))
# signal artifact latest 300 dates
for dt in common[-300:]:
 vals={}
 for s,p in px.items():
  if dt in p.index:
   i=p.index.get_loc(dt)
   if i>=65:
    vals[s]=(p.iloc[i]/p.iloc[i-10]-1,p.iloc[i]/p.iloc[i-40]-1,p.pct_change().iloc[i-20:i].std()*np.sqrt(20))
 if len(vals)>=8:
  r=np.array([v[0] for v in vals.values()]); disp=np.std(r); w=np.clip((disp-.045)/.035,0,1); med=np.median(r); med40=np.median([v[1] for v in vals.values()])
  for s,v in vals.items(): sigrows.append({'date':dt.date().isoformat(),'symbol':s,'signal':((v[0]-med)*(1-w)-.45*(v[0]-med)*w)/max(v[2],1e-5)+.2*(v[1]-med40)/max(v[2],1e-5)})
pd.DataFrame(sigrows).to_csv('scripts/miner_2_20350514_dispersion_switch_signal.csv',index=False)
print('artifact rows',len(sigrows),'common',len(common))
