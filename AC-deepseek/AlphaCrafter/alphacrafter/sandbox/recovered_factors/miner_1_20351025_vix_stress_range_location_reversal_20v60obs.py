import os, glob, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr

# One idea: stress-state range-location reversal. In high VIX environments, favor
# assets nearest their 20-observation low and penalize those near their high.
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
cl=[]; hi=[]; lo=[]
for a in assets:
    x=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
    cl.append(x['close'].rename(a)); hi.append(x['high'].rename(a)); lo.append(x['low'].rename(a))
C=pd.concat(cl,axis=1); H=pd.concat(hi,axis=1); L=pd.concat(lo,axis=1)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index()['close']
idx=C.index.intersection(v.index); C,H,L,v=C.loc[idx],H.loc[idx],L.loc[idx],v.loc[idx]
# only visible history through 2035-10-24
cut=pd.Timestamp('2035-10-24'); C,H,L,v=C.loc[:cut],H.loc[:cut],L.loc[:cut],v.loc[:cut]
range20=H.rolling(20,min_periods=20).max()-L.rolling(20,min_periods=20).min()
loc=(C-L.rolling(20,min_periods=20).min())/range20.replace(0,np.nan)
stress=(v>v.rolling(60,min_periods=60).median())
sig=-(loc-0.5).where(stress,0.0)
# Store canonical candidate signal, indexed date x symbol
out='scripts/miner_1_20351025_vix_stress_range_location_reversal_20v60obs_signal.pkl'
pickle.dump(sig,open(out,'wb'))

def metrics(h):
 f=C.shift(-h)/C-1; vals=[]; ns=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   r=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(r): vals.append((dt,r));ns.append(len(z))
 s=pd.Series(dict(vals)); mean=s.mean(); sd=s.std(ddof=1)
 return s, dict(ic=mean,icir=mean/sd if sd else np.nan,hit=(s>0).mean(),dates=len(s),n=np.mean(ns))
print('FACTOR vix_stress_range_location_reversal_20v60obs')
print('panel',C.index.min().date(),C.index.max().date(),'assets',len(assets),'stress_rate',round(stress.mean(),5),'coverage',round(sig.notna().mean().mean(),5),'active_dates',int(stress.sum()))
for h in [1,5,10,20,40]:
 s,m=metrics(h);print('H',h, {k:round(v,5) if isinstance(v,float) else v for k,v in m.items()})
 if h==20:
  for name,a,b in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_current','2034-01-01','2035-10-24')]:
   q=s.loc[a:b];print('REGIME',name,'dates',len(q),'ic',round(q.mean(),5),'icir',round(q.mean()/q.std(ddof=1),5),'hit',round((q>0).mean(),5))
# stability / turnover conditional signal
print('rank_stability',round(sig.corrwith(sig.shift(1),axis=1,method='spearman').mean(),5))
print('turnover_proxy',round((sig.rank(axis=1,pct=True)-sig.shift().rank(axis=1,pct=True)).abs().mean(axis=1).mean(),5))
# correlation evidence only resolvable pkl; contract cannot be met absent all 30
corr=[]
for p in glob.glob('scripts/*signal.pkl'):
 try:
  q=pickle.load(open(p,'rb'))
  if isinstance(q,pd.DataFrame):
   common=sig.stack().index.intersection(q.stack().index)
   if len(common)>100:
    r=spearmanr(sig.stack().loc[common],q.stack().loc[common]).statistic
    corr.append((os.path.basename(p),r))
 except Exception: pass
corr=sorted(corr,key=lambda x:abs(x[1]),reverse=True)
print('RESOLVABLE_LIBRARY_ARTIFACTS',len(corr))
print('TOP_CORRELATIONS',[(a,round(b,5)) for a,b in corr[:8]])
