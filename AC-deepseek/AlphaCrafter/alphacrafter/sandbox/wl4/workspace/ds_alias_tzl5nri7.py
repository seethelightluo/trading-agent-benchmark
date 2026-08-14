import os
for f in sorted(os.listdir('scripts')):
    if f.startswith('_batch') or 'results' in f or f.endswith('_out.txt') or f.endswith('_err.txt'):
        print(f, os.path.getsize('scripts/'+f))
