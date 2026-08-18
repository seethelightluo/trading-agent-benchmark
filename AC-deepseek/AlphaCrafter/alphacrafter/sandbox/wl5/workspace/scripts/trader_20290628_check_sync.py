import json, re

src = open('strategy.py').read()
print("strategy.py length:", len(src))
ens = json.load(open('factors/factor_ensemble.json'))
sel = [f['factor_id'] for f in ens['selected_factors']]
print("selected:", sel)
missing = [f for f in sel if f not in src]
print("missing from strategy.py:", missing)
print("has register_hook:", 'register_hook' in src)
print("MAX_W:", re.findall(r'MAX_W\s*=\s*[\d.]+', src))
print("MIN_W:", re.findall(r'MIN_W\s*=\s*[\d.]+', src))
print("ensemble as_of:", ens.get('as_of'))
print("method:", ens.get('method'))
