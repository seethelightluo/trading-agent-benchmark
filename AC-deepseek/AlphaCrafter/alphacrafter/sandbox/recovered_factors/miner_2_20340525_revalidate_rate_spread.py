from pathlib import Path
src=Path('scripts/miner_2_20330428_inverse_dispersion_amplified_us_cn_rate_spread_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2033-04-27')", "E=pd.Timestamp('2034-05-24')")
src=src.replace("20330428", "20340525")
exec(compile(src,'reval_20340525','exec'))
