from pathlib import Path
p=Path('scripts/miner_2_20300613_negative_overnight_gap_recovery_residual_20.py');s=p.read_text();s=s.replace("for n,m in [('2020_21'", "for n,m in [] # [('2020_21'")
# above syntactically leaves rest invalid. Instead replace whole regime loop line including list literal through ]:
lines=s.splitlines(); lines=["for n,m in []:" if line.startswith("for n,m in [('2020_21'") else line for line in lines];p.write_text('\n'.join(lines)+'\n')
