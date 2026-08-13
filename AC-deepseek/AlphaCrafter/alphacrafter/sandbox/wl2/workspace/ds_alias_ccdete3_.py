python -c "
import json, os, glob
# files modified recently (mtime > now-1h means after step)
import time
now = time.time()
for root in ['.', '../persistent']:
    for f in glob.glob(root + '/**/*', recursive=True):
        if os.path.isfile(f) and '__pycache__' not in f and not f.endswith('.pyc'):
            mt = os.path.getmtime(f)
            if now - mt < 7200:
                print(round(now-mt,1), 's ago', f)
" | head -30