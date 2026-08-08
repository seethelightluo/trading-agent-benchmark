from pathlib import Path
p=Path('scripts/miner_2_20300613_negative_overnight_gap_recovery_residual_20.py');s=p.read_text().replace("mx=-1\nfor n,x", "mx=-1;who='NONE';cells=0\nfor n,x");p.write_text(s)
