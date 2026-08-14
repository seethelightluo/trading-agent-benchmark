import os
# find batch4 outputs and results
for f in os.listdir('scripts'):
    if 'batch4' in f or 'batch5' in f:
        print(f, os.path.getsize('scripts/'+f))
print('---root---')
for f in os.listdir('.'):
    if 'batch' in f or 'out' in f or 'pkl' in f:
        print(f, os.path.getsize(f))
