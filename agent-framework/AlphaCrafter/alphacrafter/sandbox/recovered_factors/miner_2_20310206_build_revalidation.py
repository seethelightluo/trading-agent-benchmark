"""Refresh date in the point-in-time revalidation script."""
from pathlib import Path
s=Path('scripts/miner_2_20301114_revalidate_admitted_tail_correlation_asymmetry_residual_60.py').read_text()
s=s.replace("E=pd.Timestamp('2030-11-13')", "E=pd.Timestamp('2031-02-05')")
s=s.replace('visible through 2030-11-13', 'visible through 2031-02-05')
p=Path('scripts/miner_2_20310206_revalidate_tail_correlation_asymmetry_residual_60.py')
p.write_text(s)
print(p)
