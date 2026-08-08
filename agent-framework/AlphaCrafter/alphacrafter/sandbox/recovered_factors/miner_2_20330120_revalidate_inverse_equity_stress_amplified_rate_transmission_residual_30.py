"""Scheduled fixed-specification revalidation through 2033-01-19.
Runs the exact admitted inverse equity-stress/amplified US10Y transmission residual
specification using only data visible at the current decision date."""
from pathlib import Path
src=Path('scripts/miner_2_20320624_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("2032-06-23", "2033-01-19")
exec(compile(src, 'miner_2_equity_stress_rate_revalidation_20330120', 'exec'))
