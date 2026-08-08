"""One idea: volume-supported peer-relative downside recovery.
An asset scores highly when its volume-surprised days of peer-relative weakness
have subsequently shown consistently stronger five-session recoveries. Recovery
outcomes are shifted five sessions, so all components are known at signal time."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.get('volume'),errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);vol=pd.DataFrame(V).reindex(P.index);r=P.pct_change(); peer=r.sub(r.median(axis=1),axis=0)
# Event and subsequent five-day outcome end five days ago: known without lookahead.
vs=np.log(vol/vol.rolling(20,min_periods=15).mean()); vw=vs.rank(axis=0,pct=True)
event=peer.shift(5)<peer.shift(5).rolling(60,min_periods=40).quantile(.25)
out=P.pct_change(5).shift(0) # return ending today, starting t-5
# volume weight is likewise lagged to the event date; weighted conditional repair.
w=vw.shift(5).where(event)
den=w.rolling(80,min_periods=45).sum(); score=(w*out).rolling(80,min_periods=45).sum()/den
f=score.where((event.astype(float).rolling(80,min_periods=45).sum()>=4)&(den>0));f=f.sub(f.median(axis=1),axis=0)
cut=P.dropna(how='all').index.max()
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=P.shift(-h)/P-1;z=[];nn=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return None
 z=np.array(z);return dict(dates=len(z),ic=round(z.mean(),6),icir=round(z.mean()/z.std(ddof=1),6),hit=round((z>0).mean(),4),mean_n=round(np.mean(nn),2),min_n=min(nn))
print('FACTOR volume_supported_peer_downside_recovery_80 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(f.notna().stack().mean(),6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().stack().mean(),6))
for h in [1,5,10,20]:print('H',h,ev(h))
for name,sp in [('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',name,ev(10,sp))
# A strict correlation screen against every admitted JSON needs executable signals;
# library factor definitions are heterogeneous and no shared signal registry exists.
# This run intentionally reports the candidate's predictive evidence only; it is not eligible for persistence.
