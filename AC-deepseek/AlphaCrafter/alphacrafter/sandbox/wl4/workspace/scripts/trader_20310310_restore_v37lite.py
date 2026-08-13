"""Trader 2031-03-10: restore v37-lite strategy from safety backup.

The last trader phase stalled and main.py installed a safety-hold strategy.
The safety backup (strategy.py.safety.bak) still contains the full v37-lite
strategy: v36 base + single risk adjustment US10Y x0.70 -> x0.60 (3 cons neg
blocks -1.39%/-1.46%/-5.86% incl large print + r20 -7.24% bond selloff).
Rebuild strategy.py = v37-lite doc header + code portion (from 'import json').
"""
src = open('strategy.py.safety.bak').read()

code_start = src.find('import json\n')
assert code_start > 0, 'code portion not found in safety backup'
code = src[code_start:]

header = '''"""Trader strategy v37-lite (2031-02-24 fired, restored 2031-03-10):
v36 base + single risk adjustment US10Y x0.70 -> x0.60 only (3 consecutive
negative blocks -1.39%/-1.46%/-5.86% incl large print + r20 -7.24% bond
selloff; SPX/000688 evidence mixed after the 12-30..01-13 recovery, kept at
v36 levels). Ensemble read live from factors/factor_ensemble.json.
"""
'''

new = header + code
with open('strategy.py', 'w') as f:
    f.write(new)
print('strategy.py rebuilt; lines:', len(new.splitlines()))
print('--- first 12 lines ---')
print('\n'.join(new.splitlines()[:12]))
