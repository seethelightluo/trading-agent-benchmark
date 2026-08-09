from pathlib import Path
src=Path('scripts/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.py').read_text()
s=src.replace("2033-04-27","2034-01-04")
if s==src: raise RuntimeError('cutoff not replaced')
exec(compile(s,'reval_uscn_20340105','exec'))
