"""One candidate: severity-weighted delayed post-stress relative rebound reversal; visible cutoff 2028-05-31."""
from pathlib import Path
code=Path('scripts/miner_1_20280504_post_stress_relative_rebound_reversal_60.py').read_text()
code=code.replace("cut=pd.Timestamp('2028-05-03')", "cut=pd.Timestamp('2028-05-31')")
exec(code, globals())
# Signal at t: systemic stress observed five sessions ago.  Its severity is the distance
# below the contemporaneously observable 60-session lower-quartile threshold, divided by
# trailing median-return dispersion and capped at 3.  Invert the subsequent 3-session
# peer-relative rebound: unusually weak rebounds after more severe shocks rank highest.
threshold=m.rolling(60,min_periods=40).quantile(.25)
scale=m.rolling(60,min_periods=40).std().replace(0,np.nan)
severity=((threshold-m)/scale).clip(lower=0,upper=3).shift(5)
rel3=P.pct_change(3).sub(P.pct_change(3).median(axis=1),axis=0)
disp3=rel3.abs().median(axis=1).replace(0,np.nan)
event3=-rel3.div(disp3,axis=0).mul(severity,axis=0).where(severity>0,axis=0)
f3=event3.rolling(60,min_periods=12).mean(); f3=f3.sub(f3.median(axis=1),axis=0)
def ev3(h,span=None):
 x=f3 if span is None else f3.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 z=np.array(z); sd=z.std(ddof=1) if len(z)>1 else np.nan
 return {'dates':len(z),'ic':round(float(z.mean()),6) if len(z) else None,'icir':round(float(z.mean()/sd),6) if np.isfinite(sd) and sd else None,'hit':round(float((z>0).mean()),4) if len(z) else None,'mean_n':round(float(np.mean(ns)),2) if ns else None,'min_n':int(min(ns)) if ns else None}
print('\nCANDIDATE severity_weighted_delayed_post_stress_rebound_reversal_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f3.notna().sum().sum()),'/',f3.size,'coverage',round(float(f3.notna().stack().mean()),5),'severity_events',int((severity>0).sum()),'mean_severity',round(float(severity[severity>0].mean()),6))
for h in H: print('H',h,ev3(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-05-31'))]: print('REGIME10',n,ev3(10,s))
print('TURNOVER',round(float(f3.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
mx=0.;who='';cells=0
# Include immediately prior admitted delayed variant explicitly: S supplies the 21 older factors.
stress_delayed=m.lt(m.rolling(60,min_periods=40).quantile(.25)).shift(5)
event2=-rel3.div(disp3,axis=0).where(stress_delayed,axis=0)
f2=event2.rolling(60,min_periods=12).mean(); f2=f2.sub(f2.median(axis=1),axis=0)
S['delayed_post_stress_relative_rebound_reversal_60']=f2
for n,g in S.items():
 q=pd.concat([f3.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6))
 if abs(rho)>mx: mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',cells,'N_FACTORS',len(S))
