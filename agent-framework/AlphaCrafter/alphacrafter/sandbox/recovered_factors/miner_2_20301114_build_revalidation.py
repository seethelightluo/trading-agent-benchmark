"""Run current revalidation of admitted tail correlation asymmetry factor."""
from pathlib import Path
s=Path('scripts/miner_2_20301003_persistent_market_stress_correlation_asymmetry_residual_60.py').read_text()
s=s.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2030-11-13')")
s=s.replace("persistent_market_stress_correlation_asymmetry_residual_60", "tail_correlation_asymmetry_residual_60_REVALIDATION")
Path('scripts/miner_2_20301114_revalidate_tail_correlation_asymmetry_residual_60.py').write_text(s)
