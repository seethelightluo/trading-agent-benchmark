#!/usr/bin/env python3
"""Check all factor JSON files for validation status and metrics."""
import json
import glob
import os

# Read ensemble
with open('factors/factor_ensemble.json') as f:
    ensemble = json.load(f)
print("=== FACTOR ENSEMBLE ===")
print(json.dumps(ensemble, indent=2))
print()

# Read all factor files
factor_files = sorted(glob.glob('factors/*.json'))
for fp in factor_files:
    if 'ensemble' in fp or fp.endswith('.bak') or '/evicted/' in fp:
        continue
    try:
        with open(fp) as f:
            d = json.load(f)
    except:
        continue
    fid = d.get('factor_id', '?')
    status = d.get('validation', {}).get('status', '?')
    metrics = d.get('validation', {}).get('metrics', {})
    lv = d.get('last_validated', '?')
    ic = metrics.get('IC', '?')
    icir = metrics.get('ICIR', '?')
    cov = metrics.get('coverage', '?')
    print(f"  {fid:30s} status={status:12s} IC={str(ic):8s} ICIR={str(icir):8s} cov={str(cov):6s} last={lv}")

# Also check evicted
print("\n=== EVICTED FACTORS ===")
for fp in sorted(glob.glob('factors/evicted/*.json')):
    if fp.endswith('.reason.json'):
        continue
    try:
        with open(fp) as f:
            d = json.load(f)
    except:
        continue
    fid = d.get('factor_id', '?')
    status = d.get('validation', {}).get('status', '?')
    lv = d.get('last_validated', '?')
    print(f"  {fid:30s} status={status:12s} last={lv}")