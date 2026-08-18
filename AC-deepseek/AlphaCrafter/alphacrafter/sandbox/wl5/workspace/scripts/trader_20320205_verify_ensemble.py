"""Verify strategy.py dynamic loader picks up v25 ensemble from factor_ensemble.json."""
import json
import types

src = open('strategy.py').read()
mod = types.ModuleType('strat')
mod.__dict__['__file__'] = 'strategy.py'
import alphacrafter.sim.utils as u
mod.__dict__['get_account_dict'] = u.get_account_dict
mod.__dict__['get_stock_daily_data'] = u.get_stock_daily_data
mod.__dict__['get_index_daily_data'] = u.get_index_daily_data
mod.__dict__['rebalance_to_weights'] = u.rebalance_to_weights
mod.__dict__['register_hook'] = u.register_hook
exec(compile(src, 'strategy.py', 'exec'), mod.__dict__)

ens = mod._load_ensemble()
print('Loaded ensemble (should be v25 as_of 2032-02-04):')
tot = 0.0
for fid, w, d in ens:
    print(f'  {fid:24s} w={w:.4f} dir={d:+d}')
    tot += w
print('n =', len(ens), '| sum =', round(tot, 6))

# cross-check against factor_ensemble.json
with open('factors/factor_ensemble.json') as f:
    d = json.load(f)
sel = d['selected_factors']
print('JSON selected_factors n =', len(sel), '| as_of =', d.get('as_of'))
json_ids = {x['factor_id'] for x in sel}
loaded_ids = {fid for fid, _, _ in ens}
print('IDs match:', json_ids == loaded_ids)
from pathlib import Path
for fid in loaded_ids:
    p = Path('factors') / f'{fid}.json'
    print(f'  {fid:24s} file_exists={p.exists()}')
