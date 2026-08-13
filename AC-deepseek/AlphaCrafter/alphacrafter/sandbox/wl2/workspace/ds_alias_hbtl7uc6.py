import os
fs = [f for f in sorted(os.listdir('scripts')) if 'miner' in f and (f.endswith('.py'))]
print('\n'.join(fs[-25:]))
