"""Scheduled revalidation of miner_2 admitted rate-transmission factor; visible endpoint 2032-06-09."""
from pathlib import Path
src=Path('scripts/miner_2_20310710_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2031-07-09')", "E=pd.Timestamp('2032-06-09')")
exec(compile(src, 'miner_2_revalidate_equity_stress_rate_20320610', 'exec'))
