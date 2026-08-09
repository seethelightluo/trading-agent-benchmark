import pathlib
p=pathlib.Path('scripts/miner_2_20280504_residual_broad_drawdown_dispersion_asymmetry_60d.py')
s=p.read_text().replace('ea.rolling(60,min_periods=20)', 'ea.rolling(60,min_periods=8)').replace('es.rolling(60,min_periods=20)', 'es.rolling(60,min_periods=12)')
p.write_text(s)
