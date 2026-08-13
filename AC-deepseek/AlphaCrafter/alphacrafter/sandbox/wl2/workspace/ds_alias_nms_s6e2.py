import os
fs = [f for f in sorted(os.listdir('scripts')) if f.endswith('.py') and ('screener' in f or 'miner_1' in f or 'miner_2' in f)]
print('\n'.join(fs[-30:]))
