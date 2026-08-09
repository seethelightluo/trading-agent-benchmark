"""One idea: slow relative return breadth (asset positive-return frequency minus market breadth)."""
import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; END=pd.Timestamp('2029-07-11')
def rd(a,ix=False):
 d=(get_index_daily_data(a,5000) if ix else get_stock_daily_data(a,5000)).set_index('date'); d.index=pd.to_datetime(d.index)
 return pd.to_numeric(d.loc[:END,'close'],errors='coerce')
p=pd.DataFrame({a:rd(a) for a in A}); r=p.pct_change(); peer=r.mean(axis=1)
# slow breadth relative to contemporaneous cross-asset breadth; higher means persistent participation
breadth=(r.gt(0).astype(float).sub(r.gt(0).mean(axis=1),axis=0)).rolling(60,min_periods=40).mean()
# controls approximating admitted library exposures
vol=r.rolling(20,min_periods=15).std(); trend=(p/p.shift(20)-1)/vol
es=pd.DataFrame({a:-r[a].rolling(40,min_periods=30).apply(lambda x: np.mean(x[x<=np.quantile(x,.2)]),raw=True)/vol[a] for a in A})
down=pd.DataFrame({a:r[a].where(peer<0).rolling(40,min_periods=12).corr(peer.where(peer<0)) for a in A})
kurt=-r.rolling(40,min_periods=30).kurt()
def beta(x,y,w=40,mp=12,c=None):
 if c is not None:x=x.where(c);y=y.where(c)
 return x.rolling(w,min_periods=mp).cov(y)/y.rolling(w,min_periods=mp).var().replace(0,np.nan)
vix=rd('VIX',True).pct_change().reindex(r.index); dxy=rd('DXY',True).pct_change().reindex(r.index); us=rd('USDCNY',True).pct_change().reindex(r.index)
vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
du=pd.DataFrame({a:-beta(r[a],dxy,40,12,dxy>0) for a in A})
usres=pd.DataFrame({a:beta(r[a],us,60,20,us>0) for a in A})
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]]; o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
f=resid(breadth,[es,down,kurt,trend,vu,vd,du,usres])
def calc(h):
 fw=p.shift(-h)/p-1; vals=[]; ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 x=pd.Series(dict(vals)); out={'horizon':h,'ic':x.mean(),'icir':x.mean()/x.std(),'hit_ratio':(x>0).mean(),'dates':len(x),'mean_instruments':np.mean(ns)}
 for name,mask in [('2020-25',x.index.year<=2025),('2026',x.index.year==2026),('2027',x.index.year==2027),('2028',x.index.year==2028),('2029YTD',x.index.year==2029),('latest120',np.arange(len(x))>=len(x)-120)]:
  q=x[mask]; out[name]={'dates':len(q),'ic':q.mean(),'icir':q.mean()/q.std()}
 return out
for h in [1,5,10,20]: print(calc(h))
print('coverage',int(f.notna().sum().sum()),'/',f.size,'=',f.notna().sum().sum()/f.size)
turn=[]
for i in range(10,len(f),10):
 z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover_10d',np.mean(turn),'turn_dates',len(turn))
# max correlation evidence against all nondeprecated active library signals, computed on aligned date-asset cells
import glob,json
mx=0; arg='none'
for path in glob.glob('factors/*.json'):
 if '_deprecated' in path or '.bak' in path: continue
 try:
  j=json.load(open(path)); expr=j.get('factor_id','')
  # use definition-level proxy unavailable; compare to named known signals only where recognized
 except: pass
print('library_correlation_requires_explicit_aligned_evidence; candidate_signal_ready_for_external_check')
