"""Inspect factor library + evicted/rejected expressions for novelty audit."""
import json
from pathlib import Path

for folder in ['factors', 'factors/evicted', 'factors/rejected', 'factors/quarantine']:
    print(f"\n########## {folder} ##########")
    for p in sorted(Path(folder).glob('*.json')):
        if p.name.endswith('.reason.json') or 'signal' in p.name:
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"{p.name}: parse error {e}")
            continue
        expr = d.get('calculation', {}).get('expression', '?')
        status = d.get('validation', {}).get('status', '?')
        print(f"{d.get('factor_id', p.stem):28s} [{status:10s}] {expr[:110]}")
