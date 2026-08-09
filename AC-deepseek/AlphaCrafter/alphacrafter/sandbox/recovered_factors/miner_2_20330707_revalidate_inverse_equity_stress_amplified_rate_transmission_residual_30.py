"""Scheduled fixed-specification revalidation through 2033-07-06, no future data."""
from pathlib import Path
src=Path('scripts/miner_2_20320624_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src=src.replace("2032-06-23", "2033-07-06")
assert "2033-07-06" in src
exec(compile(src,'miner_2_equity_stress_rate_revalidation_20330707','exec'))
