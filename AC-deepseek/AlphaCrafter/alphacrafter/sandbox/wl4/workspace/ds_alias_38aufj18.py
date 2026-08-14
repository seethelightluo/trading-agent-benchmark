import os
# look for any batch4/5 files or outputs
for f in sorted(os.listdir('scripts')):
    if 'batch' in f.lower() or 'explore' in f.lower():
        print(f, os.path.getsize('scripts/'+f))
