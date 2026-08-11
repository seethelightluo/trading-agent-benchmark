import re
p = 'scripts/miner_1_20261109_explore_batchF.py'
s = open(p).read()
s = s.replace(
    'f"to={m[\'turnover_10d_rank\']:.3f} rho={corr:.3f}({key}) "',
    'f"to={m.get(\'turnover_10d_rank\')} rho={corr:.3f}({key}) "'
)
open(p, 'w').write(s)
print('patched turnover format')
