import json
lines=open('factor_library_audit.jsonl').read().strip().split('\n')
print('total audit entries:', len(lines))
for l in lines[-6:]:
    print(l)
print()
# check earliest structure
print(json.dumps(json.loads(lines[0]), indent=1)[:1200])