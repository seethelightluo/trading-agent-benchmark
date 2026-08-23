from pathlib import Path
for f in ['scripts/revalidate_all.py','scripts/miner_3_20270812_validate_ensemble_factors.py']:
    print("="*20, f)
    t = Path(f).read_text()
    print(t[:3000])