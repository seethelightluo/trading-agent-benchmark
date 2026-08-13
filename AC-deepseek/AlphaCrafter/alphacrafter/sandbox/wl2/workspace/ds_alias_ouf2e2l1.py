import json
# check existing factor definitions to understand format and what's there
d = json.load(open('factors/downbeta_spx_60.json'))
print(json.dumps(d, indent=1)[:1800])