"""One idea: continuous peer-downside beta transition (20 minus 60 sessions), independently tested."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C); r=P.pct_change(); peer=r.median(axis=1)
def cs(x): return x.sub(x.median(axis=1),axis=0)
def dbeta(x,y,w):
 z=pd.concat([x.rename('x'),y.rename('y')],axis=1).where(lambda q:q.y<0)
 return z.x.rolling(w,min_periods=max(8,w//4)).cov(z.y)/z.y.rolling(w,min_periods=max(8,w//4)).var()
# Low recent-vs-long downside beta to the broad cross-asset peer return is favorable.
b20=pd.DataFrame({a:dbeta(r[a],peer,20) for a in A}); b60=pd.DataFrame({a:dbeta(r[a],peer,60) for a in A})
x=cs(-(b20-b60)).shift(1); fw={h:P.shift(-h)/P-1 for h in [1,5,10,20]}; cut=P.dropna(how='all').index.max()
def stats(h,lo=None):
 xx=x if lo is None else x.loc[lo[0]:lo[1]]; vals=[]; ns=[]
 for d in xx.index:
  q=pd.concat([xx.loc[d],fw[h].loc[d]],axis=1).dropna()
  if len(q)>=8: vals.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 if not vals:return {}
 z=np.array(vals); return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(ns),3),'min_breadth':min(ns)}
print('FACTOR peer_downside_beta_transition_20_60','CUTOFF',cut.date(),'ASSETS',len(A))
print('CELLS',int(x.notna().sum().sum()),'/',x.size,'COVERAGE',round(x.notna().stack().mean(),6),'TURNOVER',round(x.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(x.std(axis=1).mean(),6))
for h in fw: print('H',h,stats(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',n,stats(10,p))
# Full signal-library audit uses persisted factor signal reconstructions in the previously audited canonical script, with the candidate overwritten.
exec(open('scripts/miner_3_20310626_peer_downside_beta_change_20_60.py').read().replace("cand=cs(-(db20-db60)).shift(1)","cand=cs(-(db20-db60)).shift(1)"))
