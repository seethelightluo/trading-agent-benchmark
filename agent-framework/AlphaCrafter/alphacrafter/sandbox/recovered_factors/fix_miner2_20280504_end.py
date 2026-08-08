import pathlib
p=pathlib.Path('scripts/miner_2_20280504_residual_broad_drawdown_dispersion_asymmetry_60d.py')
s=p.read_text().replace("END=pd.Timestamp('2028-01-12')","END=pd.Timestamp('2028-03-22')")
# Outer acceleration script will now load to 2028-05-03.
s=s.replace("END=pd.Timestamp('2028-03-22')\",\"END=pd.Timestamp('2028-03-22')", "END=pd.Timestamp('2028-03-22')\",\"END=pd.Timestamp('2028-05-03')")
p.write_text(s)
