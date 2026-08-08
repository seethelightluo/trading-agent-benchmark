"""Scheduled revalidation: admitted VIX-stress peer-correlation residual, current cutoff and full active-library distinctness."""
import io,contextlib,json
from pathlib import Path
import pandas as pd,numpy as np
src=Path('scripts/miner_1_20280921_vix_stress_peer_correlation_residual_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2028-09-20')", "END=pd.Timestamp('2029-03-07')")
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src,'vix_original','exec'),globals())
# f is precisely the persisted construction. Produce current metrics with granular regimes.
def summary(x):
 return {'dates':len(x),'ic':float(x.mean()),'icir':float(x.mean()/x.std()) if len(x)>1 and x.std() else None,'hit':float((x>0).mean()) if len(x) else None}
def current_metric(h):
 fw=p.shift(-h)/p-1; q=[]; nn=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8:q.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));nn.append(len(z))
 x=pd.Series(dict(q)); turns=[]
 for i in range(10,len(f),10):
  z=pd.concat([f.iloc[i-10],f.iloc[i]],axis=1).dropna()
  if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 return {'horizon':h,'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/x.std()),'hit_ratio':float((x>0).mean()),'ic_dates':len(x),'ic_se':float(x.std()/np.sqrt(len(x))),'mean_instruments':float(np.mean(nn)),'turnover_10d':float(np.mean(turns)),'regimes':{str(y):summary(x[x.index.year==y]) for y in [2026,2027,2028,2029]},'latest_120':summary(x.tail(120))}
print('REVALIDATION_VISIBLE',END.date(),'assets',len(A),'price_dates',len(p),'cells',int(f.count().sum()),'possible',f.size)
for h in [1,5,10,20]: print('METRIC',json.dumps(current_metric(h)))
# Add subsequently admitted active signals absent from original correlation reconstruction.
def resid(y,cs):
 o=y*np.nan
 for d in y.index:
  z=pd.concat([y.loc[d].rename('y')]+[q.loc[d].rename(str(i)) for i,q in enumerate(cs)],axis=1).dropna()
  if len(z)>=8:
   X=np.c_[np.ones(len(z)),z.iloc[:,1:]];o.loc[d,z.index]=z.y-X@np.linalg.lstsq(X,z.y,rcond=None)[0]
 return o
# all original reconstructed signal variants plus active additions
vix=vr; vd=pd.DataFrame({a:r[a].where(vix<0).rolling(40,min_periods=12).mean()/vol[a] for a in A}); vu=pd.DataFrame({a:r[a].where(vix>0).rolling(40,min_periods=12).mean()/vol[a] for a in A})
vdown=resid(vd,[es,downp,kurt,trend,vu]); vup=resid(vu,[es,downp,kurt,trend])
# observation-safe volume factor
rv=pd.DataFrame({a:pd.to_numeric(dat[a].volume,errors='coerce')/pd.to_numeric(dat[a].volume,errors='coerce').rolling(20,min_periods=10).mean() for a in A})
# inverse cross-asset correlation concentration: low dispersion of peer correlations is unfavorable
corrs=pd.DataFrame({a:r[a].rolling(40,min_periods=25).corr(peer[a]) for a in A}); conc=-corrs.sub(corrs.mean(axis=1),axis=0).abs()
# admitted downside beta compression residual
shortb=pd.DataFrame({a:beta(r[a],peer[a],peer[a]<0) for a in A}); longb=pd.DataFrame({a:r[a].where(peer[a]<0).rolling(60,min_periods=20).cov(peer[a].where(peer[a]<0))/peer[a].where(peer[a]<0).rolling(60,min_periods=20).var() for a in A}); comp=resid(-(shortb-longb),[trend,downp,asym,dep])
# Explicit IDs (some duplicate constructions are retained as distinct admitted records; correlations remain evidenced).
alllib=dict(lib)
alllib.update({'relative_volume_participation_20d':rv,'inverse_cross_asset_correlation_concentration_40obs':conc,'inverse_residual_downside_peer_beta_compression_20_60':comp,'orthogonal_vix_up_resilience_40obs':vup,'orthogonal_vix_down_recovery_resilience_40obs':vdown})
mx=-1;who=''
for n,x in alllib.items():
 z=pd.concat([f.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=z.f.corr(z.x,method='spearman') if len(z)>1 else np.nan
 print('LIB',n,'rho',rho,'cells',len(z))
 if not np.isfinite(rho): raise RuntimeError('missing correlation evidence '+n)
 if abs(rho)>mx: mx=abs(rho);who=n
print('MAX',mx,who,'LIBRARY_SIGNALS',len(alllib))
