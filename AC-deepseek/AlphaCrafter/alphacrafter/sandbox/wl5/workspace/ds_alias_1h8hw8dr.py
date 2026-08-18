import json
lines = open('factor_library_audit.jsonl').read().splitlines()
print("total lines:", len(lines))
# print the last 3 lines raw (first 500 chars each)
for ln in lines[-3:]:
    print(ln[:600])
    print('====')
