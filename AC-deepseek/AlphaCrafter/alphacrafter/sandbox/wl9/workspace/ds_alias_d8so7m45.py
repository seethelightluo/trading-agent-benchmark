from pathlib import Path
import json
ens = json.loads(Path('factor_ensemble.json').read_text())
ids = [f['factor_id'] for f in ens['selected_factors']]
fs = Path('factors')
print('ensemble ids:', ids)
print('all factor json present:', all((fs/f'{i}.json').exists() for i in ids))
missing = [i for i in ids if not (fs/f'{i}.json').exists()]
print('missing:', missing)
print('total factors in library:', len(list(fs.glob('*.json'))))