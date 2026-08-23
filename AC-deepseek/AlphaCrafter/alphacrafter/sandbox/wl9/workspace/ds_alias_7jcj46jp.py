from pathlib import Path
for f in ['scripts/miner_3_20270617_rel_momentum.py','scripts/miner_2_20270812_rolling_sharpe_momentum.py']:
    print("="*20, f)
    txt = Path(f).read_text()
    print(txt[:2600])