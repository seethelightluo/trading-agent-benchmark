from pathlib import Path
p=Path('scripts/miner_2_20300613_negative_overnight_gap_recovery_residual_20.py');x=p.read_text();a=x.index("for n,m in [] #");b=x.index("\n x=ics[10][m]",a);x=x[:a]+"for n,m in []:"+x[b:];p.write_text(x)
