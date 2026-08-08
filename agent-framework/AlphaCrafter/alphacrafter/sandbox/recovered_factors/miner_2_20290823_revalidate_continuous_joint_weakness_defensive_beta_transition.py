"""Timed revalidation: continuous joint-weakness-weighted residual defensive beta transition."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20290809_continuous_joint_weakness_weighted_defensive_beta_transition_60_20d.py',encoding='utf8').read()
# Original construction and full historical evaluator; only completed observations through 2029-08-22.
src=src.replace("END=pd.Timestamp('2029-08-08')", "END=pd.Timestamp('2029-08-22')")
src=src.replace("validation_end',END.date()", "revalidation_end',END.date()")
exec(src,globals())
# Explicit contemporaneous drift report on the admission 10-session horizon.
x=ics[10]
for label,mask in [('pre_2027',x.index<'2027'),('2027_2028',(x.index>='2027')&(x.index<'2029')),('2029_ytd',x.index>='2029'),('recent_120d',x.index>=x.index.max()-pd.Timedelta(days=168))]:
 q=x[mask]; sd=q.std(ddof=1)
 print('DRIFT10',label,'dates',len(q),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/sd),6),'hit',round(float((q>0).mean()),6))
print('REVALIDATION_CONTRACT horizon=10 IC',round(float(metrics[10]['daily_paper_ic']),6),'ICIR',round(float(metrics[10]['daily_paper_icir']),6),'coverage',round(float(f.notna().mean().mean()),6))
