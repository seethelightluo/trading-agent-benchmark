import json
# read the audit file to understand current factor states
import os
p='factor_library_audit.jsonl'
if os.path.exists(p):
    lines = open(p).readlines()
    print('audit lines:', len(lines))
    last = json.loads(lines[-1])
    print('last audit keys:', list(last.keys()))
    print(json.dumps(last, indent=1)[:1500])
