"""Scheduled fixed-specification revalidation: inverse dispersion-amplified US-China rate-spread transmission residual (30).
Point-in-time cutoff is the prior completed session, 2033-10-12.
"""
from pathlib import Path
src=Path('scripts/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.py').read_text()
s=src.replace("E=pd.Timestamp('2033-04-27')", "E=pd.Timestamp('2033-10-12')")
if s == src:
    raise RuntimeError('Expected cutoff literal absent; fixed-spec revalidation stopped')
exec(compile(s, 'miner_2_20331013_revalidate_us_cn_rate_spread_30', 'exec'))
