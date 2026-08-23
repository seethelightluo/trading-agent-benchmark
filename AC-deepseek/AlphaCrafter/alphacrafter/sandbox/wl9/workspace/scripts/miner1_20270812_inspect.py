#!/usr/bin/env python3
"""Show structure and validation details of key factors."""
import json

for fname in ['mom_120d_skip5', 'mom_10d_skip5', 'vix_beta_cond_60x20', 'vol_z_20d', 'ac1_120d']:
    path = f'factors/{fname}.json'
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"{fname}: error {e}")
        continue
    print(f"=== {fname} ===")
    print(f"  Keys: {list(d.keys())}")
    val = d.get('validation', {})
    print(f"  validation keys: {list(val.keys())}")
    print(f"  status: {val.get('status')}")
    metrics = val.get('metrics', {})
    print(f"  metrics: {json.dumps(metrics, indent=4)}")
    print(f"  last_validated: {d.get('last_validated', 'N/A')}")
    print()