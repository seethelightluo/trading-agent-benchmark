from pathlib import Path
src=Path('scripts/miner_2_20330707_revalidate_inverse_equity_stress_amplified_rate_transmission_residual_30.py').read_text()
s=src.replace('2033-07-06','2034-01-04')
if s==src: raise RuntimeError('cutoff not replaced')
exec(compile(s,'reval_equity_rate_20340105','exec'))
