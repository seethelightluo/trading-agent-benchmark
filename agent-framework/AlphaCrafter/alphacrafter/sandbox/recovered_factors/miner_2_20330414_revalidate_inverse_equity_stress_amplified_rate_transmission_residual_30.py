"""Scheduled fixed-specification revalidation through 2033-04-13.
Uses only the point-in-time data API and the admitted specification."""
from pathlib import Path
src=Path('scripts/miner_2_20320624_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("2032-06-23", "2033-04-13")
exec(compile(src, 'miner_2_equity_stress_rate_revalidation_20330414', 'exec'))
