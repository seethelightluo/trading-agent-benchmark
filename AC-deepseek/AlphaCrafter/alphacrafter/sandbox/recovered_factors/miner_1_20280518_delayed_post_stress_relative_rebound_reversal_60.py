"""One candidate: delayed post-stress relative rebound reversal, validated through visible 2028-05-17."""
# Reuse the exact admitted-library signal implementations from the prior validated script,
# with its data cutoff advanced solely to the current visible-data cutoff.
from pathlib import Path
code=Path('scripts/miner_1_20280504_post_stress_relative_rebound_reversal_60.py').read_text()
code=code.replace("cut=pd.Timestamp('2028-05-03')", "cut=pd.Timestamp('2028-05-17')")
exec(code, globals())
# At decision t, use only a stress flag known five observations earlier.  Measure the
# three-observation peer-relative rebound ending yesterday, and invert it: subdued
# delayed rebound after systemic stress is hypothesized to lead the next swing.
stress_delayed=m.lt(m.rolling(60,min_periods=40).quantile(.25)).shift(5)
rel3=(P/P.shift(3)-1).sub((P/P.shift(3)-1).median(axis=1),axis=0)
disp3=rel3.abs().median(axis=1).replace(0,np.nan)
event2=-rel3.div(disp3,axis=0).where(stress_delayed,axis=0)
f2=event2.rolling(60,min_periods=12).mean(); f2=f2.sub(f2.median(axis=1),axis=0)
def ev2(h,span=None):
 x=f2 if span is None else f2.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v);ns.append(len(q))
 z=np.array(z); sd=z.std(ddof=1) if len(z)>1 else np.nan
 return {'dates':len(z),'ic':round(float(z.mean()),6) if len(z) else None,'icir':round(float(z.mean()/sd),6) if np.isfinite(sd) and sd else None,'hit':round(float((z>0).mean()),4) if len(z) else None,'mean_n':round(float(np.mean(ns)),2) if ns else None,'min_n':int(min(ns)) if ns else None}
print('\nCANDIDATE delayed_post_stress_relative_rebound_reversal_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f2.notna().sum().sum()),'/',f2.size,'coverage',round(float(f2.notna().stack().mean()),5),'delayed_stress_events',int(stress_delayed.sum()))
for h in H: print('H',h,ev2(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-05-17'))]: print('REGIME10',n,ev2(10,s))
print('TURNOVER',round(float(f2.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
mx=0.;who='';cells=0
for n,g in S.items():
 q=pd.concat([f2.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',cells,'N_FACTORS',len(S))
