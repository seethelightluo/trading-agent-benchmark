"""Corrected regime supplement for miner_3 2028-06-01 revalidation."""
src=open('scripts/miner_3_20280601_revalidate_inverse_cross_asset_correlation_concentration_40obs.py').read()
src=src.replace("def calc(h,subset=None):","def calc(h,subset=None,end=None):").replace("for d in f.index if subset is None else f.index[f.index>=subset]:","for d in f.index[(f.index>=subset) & (f.index<=end)] if subset is not None else f.index:")
src=src.replace("return {n:calc(h,pd.Timestamp(s)) for n,s in {'2026':'2026-01-01','2027':'2027-01-01','2028_ytd':'2028-01-01','recent_120_sessions':str(p.index[max(0,len(p)-120)].date())}.items()}","return {n:calc(h,pd.Timestamp(s),pd.Timestamp(e)) for n,s,e in [('2026','2026-01-01','2026-12-31'),('2027','2027-01-01','2027-12-31'),('2028_ytd','2028-01-01','2028-05-31'),('recent_120_sessions',str(p.index[max(0,len(p)-120)].date()),'2028-05-31')]}")
exec(compile(src,'corrected.py','exec'))
