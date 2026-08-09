"""Revalidate one admitted factor: residual downside event-spacing relief, completed-bar cutoff 2031-10-01."""
import pathlib
src=pathlib.Path('scripts/miner_1_20310724_residual_downside_event_spacing_relief_5_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2031-07-23')", "END=pd.Timestamp('2031-10-01')")
src=src.replace('"""One idea: residual-downside event-spacing relief; 2031-07-24 completed-bar validation."""', '"""One idea: quarterly revalidation of residual-downside event-spacing relief; 2031-10-01 completed-bar cutoff."""')
pathlib.Path('scripts/miner_1_20311002_revalidate_residual_downside_event_spacing_relief_5_60obs.py').write_text(src)
