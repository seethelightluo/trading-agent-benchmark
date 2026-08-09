"""Miner_2 periodic point-in-time revalidation of the admitted inverse equity-stress rate-transmission factor.
Single-idea validation; requested cutoff is prior completed session to 2032-02-19."""
from pathlib import Path
src=Path('scripts/miner_2_20310710_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2031-07-09')", "E=pd.Timestamp('2032-02-18')")
src=src.replace('20310710', '20320219')
exec(compile(src, 'miner_2_revalidate_equity_stress_rate_20320219', 'exec'))
