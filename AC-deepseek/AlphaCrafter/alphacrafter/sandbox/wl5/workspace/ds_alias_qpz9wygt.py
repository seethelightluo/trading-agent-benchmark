import os
files = os.listdir('.')
print(len(files))
print([f for f in files if 'factor' in f or 'ensemble' in f or 'memory' in f])
print(os.path.exists('factor_ensemble.json'), os.path.exists('memory.txt'))