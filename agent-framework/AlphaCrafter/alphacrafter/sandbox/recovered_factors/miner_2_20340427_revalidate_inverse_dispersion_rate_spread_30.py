"""Revalidate admitted inverse dispersion-amplified US-CN rate spread factor through 2034-04-26."""
from pathlib import Path
src=Path('scripts/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2033-04-27')", "E=pd.Timestamp('2034-04-26')")
src=src.replace("20330428", "20340427")
exec(compile(src,'inverse_dispersion_rate_spread_revalidation_20340427','exec'))
