"""Sanity check: reproduce mom_10d_skip5 validation with the shared framework."""
import sys
sys.path.insert(0, "scripts")
from miner2_20260730_factorlib import load_panel, factor_panel, fwd_ret_panel, validate

P = load_panel()

def mom10_skip5(s):
    return s.shift(5) / s.shift(15) - 1.0

f = factor_panel(P, mom10_skip5)
fwd10 = fwd_ret_panel(P, 10)
res = validate(f, fwd10, label="mom_10d_skip5", expected_dir=1)
import json
print(json.dumps(res, indent=1))
