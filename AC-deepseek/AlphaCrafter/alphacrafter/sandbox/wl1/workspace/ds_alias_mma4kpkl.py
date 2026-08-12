import json
# look at most recent miner2 scripts & results
import os
for f in sorted(os.listdir('scripts')):
    if 'miner2' in f and ('202812' in f or '202901' in f or '202902' in f):
        print(f)