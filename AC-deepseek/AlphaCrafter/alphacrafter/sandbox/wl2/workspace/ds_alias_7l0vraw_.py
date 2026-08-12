import json, glob
# any results file in scripts for 2028?
for f in sorted(glob.glob('scripts/*.json')):
    import os
    print(f, os.path.getsize(f))
