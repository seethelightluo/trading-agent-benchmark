"""Scheduled fixed-specification revalidation of admitted US-China rate-spread factor.
Uses only completed observations through the runtime decision cutoff; one factor only.
"""
from pathlib import Path
src=Path('scripts/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2033-04-27')", "E=pd.Timestamp('2033-07-20')")
assert "2033-07-20" in src
exec(compile(src, 'miner_2_us_cn_rate_spread_revalidation_20330721', 'exec'))
