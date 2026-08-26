import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2035-06-24'); px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index(); px[s]=d.loc[d.index<=cut]
common=sorted(set.intersection(*[set(x.index) for x in px.values()])); rec=[]; sig=[]
# volume-confirmed medium-term continuation: 20d return/vol, weighted by relative volume, winsorized
for dt in common:
 vals={}; fw={}
 for s,d in px.items():
  i=d.index.get_loc(dt)
  if i<45 or i+20>=len(d): continue
  c=d.close.astype(float); v=d.volume.astype(float).replace(0,np.nan)
  r20=c.iloc[i]/c.iloc[i-20]-1; vol=c.pct_change().iloc[i-19:i+1].std()
  vr=v.iloc[i-4:i+1].mean()/(v.iloc[i-24:i-4].mean()+1e-12)
  if not np.isfinite(vol) or vol<=0 or not np.isfinite(vr): continue
  vals[s]=r20/vol*np.sqrt(max(0.25,min(vr,4.0)))
  fw[s]={h:c.iloc[i+h]/c.iloc[i]-1 for h in [1,5,10,20]}
 if len(vals)<8: continue
 a=np.array(list(vals.values())); lo,hi=np.nanpercentile(a,[5,95]); a=np.clip(a,lo,hi); f=dict(zip(vals,a))
 for h in [1,5,10,20]: rec.append((dt,h,spearmanr(list(f.values()),[fw[s][h] for s in f]).statistic,len(f)))
 for s in f: sig.append({'date':dt.date().isoformat(),'symbol':s,'signal':float(f[s])})
r=pd.DataFrame(rec,columns=['date','h','ic','n']); print('factor=volume_confirmed_trend cut',cut.date(),'dates',r.date.nunique(),'assets',len(U))
for h in [1,5,10,20]:
 q=r[r.h==h].ic.dropna(); print('H%d dates %d avgN %.2f IC %.6f dailyICIR %.6f hit %.3f coverage %.3f'%(h,len(q),r[r.h==h].n.mean(),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),r[r.h==h].n.sum()/(len(q)*15)))
for w in [252,756,1260]:
 q=r[r.h==20].tail(w).ic.dropna(); print('RECENT',w,'IC',q.mean(),'dailyICIR',q.mean()/q.std(ddof=1),'dates',len(q))
pd.DataFrame(sig).to_csv('scripts/miner_2_20350625_volume_confirmed_trend_signal.csv',index=False)
