"""One candidate: commonality-shock gated peer-relative trend, visible cutoff 2028-06-14."""
from pathlib import Path
code=Path('scripts/miner_1_20280504_post_stress_relative_rebound_reversal_60.py').read_text()
code=code.replace("cut=pd.Timestamp('2028-05-03')", "cut=pd.Timestamp('2028-06-14')")
exec(code,globals())
# A sharp transition to high cross-asset commonality can distinguish durable leaders from
# transient moves.  On these dates, retain each asset's 5-session peer-relative return,
# scaled by its own recent volatility; average the sparse observations over 60 sessions.
broad=corr20.median(axis=1)
q75=broad.rolling(60,min_periods=40).quantile(.75)
gate=(broad>q75) & (broad.diff(5)>0)
rel5=P.pct_change(5).sub(P.pct_change(5).median(axis=1),axis=0)
score=rel5.div(vol.replace(0,np.nan)).where(gate,axis=0)
f3=score.rolling(60,min_periods=12).mean();f3=f3.sub(f3.median(axis=1),axis=0)
def ev3(h,span=None):
 x=f3 if span is None else f3.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[];ns=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z); sd=z.std(ddof=1) if len(z)>1 else np.nan
 return {'dates':len(z),'ic':round(float(z.mean()),6) if len(z) else None,'icir':round(float(z.mean()/sd),6) if np.isfinite(sd) and sd else None,'hit':round(float((z>0).mean()),4) if len(z) else None,'mean_n':round(float(np.mean(ns)),2) if ns else None,'min_n':int(min(ns)) if ns else None}
print('\nCANDIDATE commonality_shock_gated_peer_relative_trend_60 cutoff',cut.date(),'assets',len(A))
print('CELLS',int(f3.notna().sum().sum()),'/',f3.size,'coverage',round(float(f3.notna().stack().mean()),5),'gate_events',int(gate.sum()),'mean_broad_corr',round(float(broad.mean()),6))
for h in H:print('H',h,ev3(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_28',('2027-01-01','2028-06-14'))]:print('REGIME10',n,ev3(10,s))
print('TURNOVER',round(float(f3.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# Include both post-stress factors admitted after the library template was created.
stress1=m.lt(m.rolling(60,min_periods=40).quantile(.25)).shift(1)
event1=-r.sub(m,axis=0).div(r.sub(m,axis=0).abs().median(axis=1).replace(0,np.nan),axis=0).where(stress1,axis=0)
S['post_stress_relative_rebound_reversal_60']=event1.rolling(60,min_periods=12).mean().sub(event1.rolling(60,min_periods=12).mean().median(axis=1),axis=0)
stress5=m.lt(m.rolling(60,min_periods=40).quantile(.25)).shift(5)
event5=-rel5.div(rel5.abs().median(axis=1).replace(0,np.nan),axis=0).where(stress5,axis=0)
S['delayed_post_stress_relative_rebound_reversal_60']=event5.rolling(60,min_periods=12).mean().sub(event5.rolling(60,min_periods=12).mean().median(axis=1),axis=0)
mx=0.;who='';cells=0
for n,g in S.items():
 q=pd.concat([f3.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',cells,'N_FACTORS',len(S))
