"""Scheduled fixed-specification revalidation through the last completed session, 2032-10-13.
Uses the existing complete reconstructed admitted-library screen and the exact admitted
rate-transmission formula; no post-endpoint rows are read."""
from pathlib import Path
src=Path('scripts/miner_2_20320624_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("2032-06-23", "2032-10-13")
exec(compile(src, 'miner_2_equity_stress_rate_revalidation_20321014', 'exec'))
