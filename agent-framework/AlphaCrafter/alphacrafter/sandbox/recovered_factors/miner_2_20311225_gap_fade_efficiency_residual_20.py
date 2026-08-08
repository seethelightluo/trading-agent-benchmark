"""Miner_2 single-idea validation: gap-fade efficiency residual, point-in-time through 2031-12-24.
This uses open/high/low/close only and compares the candidate to reconstructed admitted-library signals.
"""
from pathlib import Path
src=Path('scripts/miner_2_20311127_gap_fade_efficiency_residual_20.py').read_text()
src=src.replace("E=pd.Timestamp('2031-11-26')", "E=pd.Timestamp('2031-12-24')")
src=src.replace("Miner_2: downside close-location and participation confirmed recovery residual (20 observations), visible through 2030-05-29.", "Miner_2: gap-fade efficiency residual (20 observations), visible through 2031-12-24.")
# Make the reported 10d regime partition current and label the factor consistently.
src=src.replace("'2028_ytd'", "'2028_31'")
Path('scripts/miner_2_20311225_gap_fade_efficiency_residual_20.py').write_text(src)
print('wrote scripts/miner_2_20311225_gap_fade_efficiency_residual_20.py')
exec(compile(src,'gap_fade_efficiency_20311225','exec'))
