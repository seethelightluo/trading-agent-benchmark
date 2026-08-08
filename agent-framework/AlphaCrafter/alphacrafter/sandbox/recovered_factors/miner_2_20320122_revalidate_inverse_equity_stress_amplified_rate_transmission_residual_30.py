"""Revalidate miner_2's admitted inverse equity-stress rate-transmission factor.
Point-in-time cutoff 2032-01-21; retains the established single factor definition.
"""
from pathlib import Path
src = Path('scripts/miner_2_20311016_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
src = src.replace("E=pd.Timestamp('2031-10-15')", "E=pd.Timestamp('2032-01-21')")
src = src.replace("miner_2_equity_stress_rate_transmission_20311016", "miner_2_equity_stress_rate_transmission_20320122")
exec(compile(src, 'miner_2_equity_stress_rate_transmission_20320122', 'exec'))
