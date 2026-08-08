from pathlib import Path
s=Path('scripts/miner_2_20300725_tail_correlation_asymmetry_residual_60_full_library_repair.py').read_text()
s=s.replace("E=pd.Timestamp('2030-07-24')", "E=pd.Timestamp('2030-11-13')")
s=s.replace('visible through 2030-07-24', 'visible through 2030-11-13')
Path('scripts/miner_2_20301114_revalidate_admitted_tail_correlation_asymmetry_residual_60.py').write_text(s)
