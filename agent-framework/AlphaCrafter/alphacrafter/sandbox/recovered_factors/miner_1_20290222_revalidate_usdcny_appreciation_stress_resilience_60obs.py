"""Scheduled revalidation only: USDCNY appreciation-stress resilience residual (60 observations).
Runs the original single-factor methodology at the current visible cutoff without altering its construction."""
from pathlib import Path
src = Path('scripts/miner_1_20281005_usdcny_appreciation_stress_resilience_residual_60obs.py').read_text()
src = src.replace("END=pd.Timestamp('2028-10-04')", "END=pd.Timestamp('2029-02-21')")
src = src.replace("('2028_ytd',x.index.year==2028)", "('2028',x.index.year==2028),('2029_ytd',x.index.year==2029)")
exec(compile(src, 'revalidated_usdcny_resilience_source', 'exec'))
