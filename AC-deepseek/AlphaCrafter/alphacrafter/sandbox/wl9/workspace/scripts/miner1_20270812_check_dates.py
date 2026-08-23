#!/usr/bin/env python3
"""Check detailed validation timestamps and structures for all factors."""
import json, glob

for fp in sorted(glob.glob('factors/*.json')):
    if 'ensemble' in fp or '/evicted/' in fp or fp.endswith('.bak'):
        continue
    try:
        with open(fp) as f:
            d = json.load(f)
    except:
        continue
    fid = d.get('factor_id', '?')
    val = d.get('validation', {})
    lv = val.get('last_validated', 'N/A')
    pd = val.get('period', 'N/A')
    print(f"{fid:30s} last_validated={lv:25s} period={pd}")

print("\n=== Ensemble factors check ===")
with open('factors/factor_ensemble.json') as f:
    ens = json.load(f)
for sf in ens.get('selected_factors', []):
    fid = sf['factor_id']
    path = f'factors/{fid}.json'
    try:
        with open(path) as f2:
            d = json.load(f2)
        print(f"{fid:30s} status={d['validation']['status']}")
    except:
        print(f"{fid:30s} NOT FOUND in factors/")

# Check if vol_of_vol20x60 exists in main dir or only evicted
print("\n=== vol_of_vol20x60 locations ===")
for fp in glob.glob('factors/**/vol_of_vol20x60*', recursive=True):
    print(f"  {fp}")