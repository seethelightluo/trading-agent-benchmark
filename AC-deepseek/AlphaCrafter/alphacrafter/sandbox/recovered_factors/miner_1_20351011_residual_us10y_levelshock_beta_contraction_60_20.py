import os,json,glob,re
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
# One factor idea: residual US10Y shock-beta contraction (60d vs 20d). Yield shocks
# are level differences (not percentage returns), standardized with only trailing data.
assets=list(get_account_dict()['watch_list']); N=5000
def load(a):
 x=get_stock_daily_data(a,N).copy();x.date=pd.to_datetime(x.date);return x.set_index('date').sort_index().close
close=pd.DataFrame({a:load(a) for a in assets}).sort_index(); ret=close.pct_change(); market=ret.mean(axis=1)
bm=ret.rolling(60,min_periods=42).cov(market).div(market.rolling(60,min_periods=42).var(),axis=0); resid=ret.sub(bm.mul(market,axis=0))
y=close.US10Y.diff(); z=y.sub(y.rolling(60,min_periods=42).mean()).div(y.rolling(60,min_periods=42).std())
b20=resid.rolling(20,min_periods=16).cov(z).div(z.rolling(20,min_periods=16).var(),axis=0);b60=resid.rolling(60,min_periods=42).cov(z).div(z.rolling(60,min_periods=42).var(),axis=0)
raw=b60-b20; mom=resid.rolling(20,min_periods=16).mean(); semi=resid.where(resid<0).pow(2).rolling(20,min_periods=16).mean().pow(.5)
sig=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for d in sig.index:
 q=pd.concat([raw.loc[d].rename('raw'),mom.loc[d].rename('mom'),semi.loc[d].rename('semi')],axis=1).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q[['mom','semi']].to_numpy()];sig.loc[d,q.index]=q.raw-X@np.linalg.lstsq(X,q.raw.to_numpy(),rcond=None)[0]
def calc(h,lo=None,hi=None):
 fw=close.shift(-h).div(close)-1; vals=[]
 for d in sig.loc[lo:hi].index:
  q=pd.concat([sig.loc[d].rename('x'),fw.loc[d].rename('f')],axis=1).dropna()
  if len(q)>=8: vals.append(q.x.corr(q.f,method='spearman'))
 v=np.array(vals);return len(v),v.mean() if len(v) else np.nan,v.mean()/v.std(ddof=1) if len(v)>1 and v.std(ddof=1) else np.nan,(v>0).mean() if len(v) else np.nan
print('FACTOR residual_us10y_levelshock_beta_contraction_60_20');print('cutoff',close.index.max().date(),'assets',len(assets),'signal_dates',sig.notna().any(axis=1).sum(),'valid_cells',sig.notna().sum().sum(),'coverage',sig.notna().mean().mean(),'mean_names',sig.notna().sum(axis=1).mean())
for h in [1,5,10,20]:print('H',h,calc(h))
rank=sig.rank(axis=1,pct=True);zs=sig.sub(sig.mean(axis=1),axis=0).div(sig.std(axis=1),axis=0);print('turnover',rank.diff().abs().mean(axis=1).mean(),'concentration',zs.abs().stack().mean())
for lo,hi,n in [('2020-01-01','2024-12-31','2020_24'),('2025-01-01','2029-12-31','2025_29'),('2030-01-01','2034-12-31','2030_34'),('2035-01-01','2100-01-01','2035_ytd')]:print('REGIME',n,'H5',calc(5,lo,hi))
# Strict evidence: each effective factor must map to an actual readable signal panel with >=100 common cells.
def norm(x):return re.sub('[^a-z0-9]','',x.lower())
active=[]
for f in glob.glob('factors/*.json'):
 try:
  d=json.load(open(f));
  if d.get('validation',{}).get('status')=='EFFECTIVE':active.append(d.get('factor_id',''))
 except:pass
out=[];missing=[];panels=glob.glob('scripts/*signal.pkl')
for fid in active:
 toks=[t for t in re.split('[_-]',fid) if len(t)>=5 and not t.isdigit()]; scores=[(sum(norm(t) in norm(p) for t in toks),p) for p in panels]; best=max(scores) if scores else (0,'')
 if best[0]<max(2,len(toks)//2):missing.append(fid);continue
 try:
  o=pd.read_pickle(best[1]);q=pd.concat([sig.stack().rename('x'),o.stack().rename('y')],axis=1).dropna()
  if len(q)<100:missing.append(fid+' insufficient');continue
  out.append((abs(q.x.corr(q.y,method='spearman')),fid,len(q),os.path.basename(best[1])))
 except:missing.append(fid+' unreadable')
print('LIBRARY active',len(active),'resolved',len(out),'missing',len(missing));print('LIBRARY max',max(out) if out else None);print('LIBRARY top5',sorted(out,reverse=True)[:5]);print('LIBRARY missing',missing)
sig.to_pickle('scripts/miner_1_20351011_residual_us10y_levelshock_beta_contraction_60_20_signal.pkl')
