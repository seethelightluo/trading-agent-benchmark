"""Scheduled revalidation of one factor: continuous joint-weakness-weighted residual defensive beta transition."""
import numpy as np
src=open('scripts/miner_2_20290823_revalidate_continuous_joint_weakness_defensive_beta_transition.py',encoding='utf8').read()
# At the 2029-09-06 decision, 2029-09-05 is the last fully observable bar.
src=src.replace("END=pd.Timestamp('2029-08-22')", "END=pd.Timestamp('2029-09-05')")
src=src.replace("revalidation_end',END.date()", "revalidation_end_20290905',END.date()")
exec(src,globals())
# Most-recent completed 120 calendar days, reported independently for drift control.
x=ics[10]
q=x[x.index>=x.index.max()-__import__('pandas').Timedelta(days=168)]
print('LATEST_DRIFT10 dates',len(q),'IC',round(float(q.mean()),6),'ICIR',round(float(q.mean()/q.std(ddof=1)),6),'hit',round(float((q>0).mean()),6))
