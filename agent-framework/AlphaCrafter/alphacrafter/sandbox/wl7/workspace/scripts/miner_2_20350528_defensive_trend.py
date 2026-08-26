import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-05-27'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut,'close'].astype(float)
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rec=[]; sigout=[]
for dt in common:
 vals={}; fw={}
 for s,p in px.items():
  i=p.index.get_loc(dt)
  if i<65 or i+20>=len(p): continue
  r20=p.iloc[i]/p.iloc[i-20]-1; r60=p.iloc[i]/p.iloc[i-60]-1
  rets=p.pct_change().iloc[i-20:i]
  down=np.sqrt(np.mean(np.minimum(rets,0)**2))*np.sqrt(252)
  vol=rets.std()*np.sqrt(252)
  vals[s]=(r20,r60,down,vol); fw[s]={h:p.iloc[i+h]/p.iloc[i]-1 for h in [1,5,10,20]}
 if len(vals)<8: continue
 # Defensive trend: medium trend, penalized by downside risk; cross-sectional rank-like standardization
 a20=np.array([v[0] for v in vals.values()]); a60=np.array([v[1] for v in vals.values()]); dd=np.array([v[2] for v in vals.values()])
 def z(a): return (a-np.median(a))/(np.median(np.abs(a-np.median(a)))+1e-5)
 # reward persistent trend, explicitly prefer lower downside risk
 z20=dict(zip(vals, z(np.array([v[0] for v in vals.values()])))); z60=dict(zip(vals, z(np.array([v[1] for v in vals.values()])))); zd=dict(zip(vals, z(np.array([v[2] for v in vals.values()])))); f={s:0.65*z60[s]+0.35*z20[s]-0.25*zd[s] for s in vals}
 for h in [1,5,10,20]:
  q=[spearmanr([f[s] for s in f],[fw[s][h] for s in f]).statistic]
  rec.append((dt,h,q[0],len(f)))
 for s,v in vals.items(): sigout.append({'date':dt.date().isoformat(),'symbol':s,'signal':f[s]})
r=pd.DataFrame(rec,columns=['date','h','ic','n']); print('factor=defensive_persistent_trend cut',cut.date())
for h in [1,5,10,20]:
 x=r[r.h==h].ic.dropna(); print('H%d dates %d avgN %.2f IC %.6f ICIR %.6f hit %.3f coverage %.3f'%(h,len(x),r[r.h==h].n.mean(),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),r[r.h==h].n.sum()/(len(x)*15)))
for a,b in [('2020-01-01','2026-12-31'),('2027-01-01','2030-12-31'),('2031-01-01','2034-12-31'),('2035-01-01','2035-05-27')]:
 x=r[(r.h==20)&(r.date.between(a,b))].ic.dropna(); print('REG',a,b,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
pd.DataFrame(sigout).to_csv('scripts/miner_2_20350528_defensive_trend_signal.csv',index=False)
