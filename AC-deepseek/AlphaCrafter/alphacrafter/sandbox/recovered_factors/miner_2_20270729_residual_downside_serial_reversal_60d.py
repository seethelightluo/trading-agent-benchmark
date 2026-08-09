"""Miner 2: validate sign-reversed residual downside serial persistence."""
import pathlib
src=pathlib.Path('scripts/miner_2_20270715_residual_downside_serial_persistence_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-07-15')", "END=pd.Timestamp('2027-07-29')")
src=src.replace("f=pd.DataFrame({a: neg[a].rolling(60,min_periods=45).corr(neg[a].shift(1)) for a in A})", "f=-pd.DataFrame({a: neg[a].rolling(60,min_periods=45).corr(neg[a].shift(1)) for a in A})")
src=src.replace("residual_downside_serial_persistence_60d", "residual_downside_serial_reversal_60d")
pathlib.Path('scripts/miner_2_20270729_residual_downside_serial_reversal_60d.py').write_text(src)
print('written')
