"""Miner_2 correction/revalidation: smooth inverse equity-stress rate-transmission residual.
Complete library correlation evidence includes the immediately preceding admitted clipped signal.
"""
from pathlib import Path
src=Path('scripts/miner_2_20310501_exact_librarycheck.py').read_text()
src=src.replace("E=pd.Timestamp('2031-04-30')", "E=pd.Timestamp('2031-05-14')")
src=src.replace("for nm,z,orient in [('dxy','DXY',1)", "L['inverse_equity_stress_amplified_rate_transmission_residual_30']=old_factor\nfor nm,z,orient in [('dxy','DXY',1)")
src=src.replace("FACTOR smooth_equity_stress_rate_transmission_residual_30", "FACTOR smooth_equity_stress_rate_transmission_residual_30_complete_library")
exec(compile(src,'miner_2_20310515_smooth_complete_library','exec'))
