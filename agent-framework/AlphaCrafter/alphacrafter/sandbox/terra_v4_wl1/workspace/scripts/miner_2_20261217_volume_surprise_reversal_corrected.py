import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for a in A:
 d=get_stock_daily_data(a,days=4000)
 if d is not None and len(d)>0:
  z=d.set_index('date'); px[a]=pd.to_numeric(z.close,errors='coerce'); vol[a]=pd.to_numeric(z.volume,errors='coerce')
p=pd.concat(px,axis=1).sort_index(); vv=pd.concat(vol,axis=1).reindex(p.index)
# Do not forward-fill prices or volume across missing asset histories. Compute each asset's
# abnormal volume against its own trailing median; signal uses only completed t bar and is
# evaluated against t+1... return. A zero/nonfinite volume observation is invalid.
vs=vv.where(vv>0).div(vv.where(vv>0).rolling(20,min_periods=10).median()).sub(1.0)
r3=p.pct_change(3)
f=(-r3*vs.clip(-2,2)).replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(p)-10):
 for h in [1,5,10]:
  q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   rows.append((p.index[i],h,len(q),q.f.corr(q.y)))
df=pd.DataFrame(rows,columns=['date','h','n','ic'])
for h in [1,5,10]:
 q=df[df.h==h]; x=q.ic.to_numpy();
 # rank turnover, computed on adjacent valid signal dates with >=8 names
 turns=[]
 for j in range(1,len(p)):
  a=f.iloc[j-1].dropna(); b=f.iloc[j].dropna(); common=a.index.intersection(b.index)
  if len(common)>=8: turns.append((a[common].rank()!=b[common].rank()).mean())
 print({'h':h,'dates':len(x),'avgN':float(q.n.mean()) if len(q) else 0,'IC':float(np.nanmean(x)),'ICIR':float(np.nanmean(x)/np.nanstd(x,ddof=1)),'hit':float(np.mean(x>0)),'coverage':float(q.n.mean()/len(A)),'turnover':float(np.mean(turns))})
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=df[(df.h==1)&(df.date.astype(str)>=lo)&(df.date.astype(str)<=hi+'-12-31')];print('regime',lo,hi,'dates',len(q),'IC',float(q.ic.mean()) if len(q) else None)
print('assets',len(px),'period',p.index.min().date(),p.index.max().date())
# Persist a complete provenance signal artifact for any admitted candidate; this script itself
# only validates and does not claim admission.
f.stack().rename('signal').to_csv('scripts/miner_2_20261217_volume_surprise_reversal_signal.csv',header=True)
