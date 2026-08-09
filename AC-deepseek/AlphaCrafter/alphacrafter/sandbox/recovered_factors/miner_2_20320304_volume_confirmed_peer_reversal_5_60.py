"""One idea: volume-confirmed peer reversal after short-horizon moves."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
assets=get_account_dict()['watch_list']; C={}; V={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce')
p=pd.DataFrame(C); vol=pd.DataFrame(V).reindex(p.index); r=p.pct_change(); r5=p.pct_change(5)
# Reversal is assigned more strongly where the five-day move was accompanied by
# unusual own-volume participation, a simple exhaustion/overreaction proxy.
vr=np.log(vol.rolling(5,min_periods=4).mean()/vol.rolling(60,min_periods=40).mean())
vr=vr.clip(vr.quantile(.05,axis=1),vr.quantile(.95,axis=1),axis=0)
rel=r5.sub(r5.median(axis=1),axis=0)
raw=-rel*(1+vr.clip(lower=-.75,upper=1.5))
sig=raw.clip(raw.quantile(.05,axis=1),raw.quantile(.95,axis=1),axis=0).shift(1)
fwd={h:p.shift(-h).div(p).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=sig.loc[lo:hi] if lo else sig; z=[];b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);b.append(len(q))
 z=np.array(z)
 return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(b),3),'min_breadth':min(b)}
cut=p.dropna(how='all').index.max(); rank=sig.rank(axis=1,pct=True)
print('FACTOR volume_confirmed_peer_reversal_5_60 CUTOFF',cut.date(),'ASSETS',len(assets))
print('CELLS',int(sig.notna().sum().sum()),'/',sig.size,'COVERAGE',round(sig.notna().stack().mean(),6),'TURNOVER',round(rank.diff().abs().stack().mean(),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cut.date())),('recent180',str(cut-pd.Timedelta(days=180)),str(cut.date()))]:print('REGIME5',n,stat(5,lo,hi))
for n,x in {'peer_reversal_5':-r5,'relative_volume':vr,'simple_momentum_20':p.pct_change(20)}.items():
 q=pd.concat([sig.stack(),x.shift(1).stack()],axis=1).dropna();print('PROXY',n,'cells',len(q),'rho',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6))
