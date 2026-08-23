"""Compact factor library status view (metadata only, no signal decode)."""
import json, glob, os

for fp in sorted(glob.glob('factors/*.json')):
    if 'bak' in fp or os.path.isdir(fp):
        continue
    try:
        d = json.load(open(fp))
        v = d.get('validation', {})
        m = v.get('metrics', {})
        print(f"{d.get('factor_id','?'):28s} status={v.get('status','?'):12s} ic={m.get('ic')} icir={m.get('icir')} "
              f"cov={m.get('coverage_dates_ge8')} last_val={v.get('last_validated')} dir={d.get('expected_direction')}")
    except Exception as e:
        print(fp, 'ERR', e)