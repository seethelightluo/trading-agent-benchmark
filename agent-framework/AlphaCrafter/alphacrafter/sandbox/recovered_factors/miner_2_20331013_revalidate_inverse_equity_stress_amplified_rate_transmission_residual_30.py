"""Scheduled fixed-specification revalidation through 2033-10-12, no future data."""
from pathlib import Path
src=Path('scripts/miner_2_20330707_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
s=src.replace("2033-07-06", "2033-10-12")
if s == src: raise RuntimeError('cutoff replacement absent')
exec(compile(s,'miner_2_equity_stress_rate_revalidation_20331013','exec'))
