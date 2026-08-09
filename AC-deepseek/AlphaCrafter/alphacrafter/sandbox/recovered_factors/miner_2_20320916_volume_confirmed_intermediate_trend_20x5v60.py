# miner_2: volume-confirmed intermediate trend, validated as a single interpretable idea
import os, glob, pickle, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']; close=[]; vol=[]
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.set_index('date')
    close.append(d['close'].rename(a))
    # Use reported volume only; no imputation so coverage is explicit.
    vol.append(d['volume'].rename(a) if 'volume' in d else pd.Series(index=d.index,name=a,dtype=float))
px=pd.concat(close,axis=1).sort_index(); volume=pd.concat(vol,axis=1).sort_index()
# 20-session log trend, confirmed/dampened by current 5d participation relative to
# its preceding 60d norm. log transform limits extreme volumes.
trend=np.log(px/px.shift(20))
participation=np.log1p(volume.rolling(5,min_periods=5).mean()/volume.rolling(60,min_periods=40).mean())
sig=(trend*participation).replace([np.inf,-np.inf],np.nan)
sig=sig.where(sig.count(axis=1)>=8)
print('CANDIDATE volume_confirmed_intermediate_trend_20x5v60')
print('panel_dates',int(sig.notna().any(axis=1).sum()),'assets',len(assets),'coverage',float(sig.notna().mean().mean()),'mean_instruments',float(sig.count(axis=1).mean()))
def stats(s,h):
 fw=np.log(px.shift(-h)/px); ic=[]; n=[]
 for dt in s.index:
  z=pd.concat([s.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   x=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(x): ic.append(x); n.append(len(z))
 x=np.array(ic,float)
 if len(x)<2:return {'ic':None,'icir':None,'hit':None,'dates':len(x),'mean_n':None}
 return {'ic':float(x.mean()),'icir':float(x.mean()/(x.std(ddof=1)+1e-12)),'hit':float((x>0).mean()),'dates':len(x),'mean_n':float(np.mean(n))}
for h in (1,5,10,20): print('H',h,stats(sig,h))
for name,a,b in [('2020_2023','2020-01-01','2023-12-31'),('2024_2027','2024-01-01','2027-12-31'),('2028_2030','2028-01-01','2030-12-31'),('2031_2032','2031-01-01','2032-12-31'),('recent_6m','2032-03-01','2032-12-31')]: print('REGIME',name,stats(sig.loc[a:b],10))
stab=sig.corrwith(sig.shift(),axis=1,method='spearman').mean(); print('rank_stability',float(stab),'turnover',float(1-stab))
out='scripts/miner_2_20320916_volume_confirmed_intermediate_trend_20x5v60_signal.pkl'; pickle.dump(sig,open(out,'wb'))
# Complete audit is binding: enumerate active JSON factors, then inspect whether an artifact exists.
active=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(os.path.basename(fn))
 except: pass
base=sig.stack().rename('candidate'); found=[]; missing=[]; mx=-1; who=None
for f in active:
 stem=os.path.splitext(f)[0]; choices=glob.glob('scripts/*'+stem.split('_',2)[-1]+'*_signal.pkl')
 if not choices: missing.append(f); continue
 try:
  q=pickle.load(open(choices[-1],'rb')); q=q.stack() if isinstance(q,pd.DataFrame) else q
  z=pd.concat([base,q.rename('other')],axis=1).dropna()
  if len(z)<100: missing.append(f); continue
  c=abs(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); found.append(f)
  if c>mx: mx,who=c,f
 except: missing.append(f)
print('LIBRARY_AUDIT effective',len(active),'evidence',len(found),'missing',len(missing),'max_abs_corr',mx,'best',who)
print('MISSING',','.join(missing)); print('cutoff',px.index.max().date())
