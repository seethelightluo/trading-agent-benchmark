"""One candidate: residual drawdown-conditioned upside-capture acceleration (20d vs 60d), re-test to current cutoff."""
import pathlib
src=pathlib.Path('scripts/miner_2_20281130_residual_drawdown_conditioned_upside_capture_acceleration_20_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2028-11-29')", "END=pd.Timestamp('2029-04-04')")
exec(src)
