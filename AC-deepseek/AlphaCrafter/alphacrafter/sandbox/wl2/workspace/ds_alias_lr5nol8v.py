import json
# check audit file structure
with open('factor_library_audit.jsonl') as f:
    lines = f.readlines()
print('total audit lines:', len(lines))
for line in lines[-3:]:
    d = json.loads(line)
    print(json.dumps(d, indent=1)[:800])
    print('====')
