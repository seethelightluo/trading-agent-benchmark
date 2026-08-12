import json
d = json.load(open('scripts/miner2_screen_all_v2_results.json'))
for x in d:
    if x.get('passed'):
        print(json.dumps({k:v for k,v in x.items() if k!='extra'}, indent=1)[:900])
        print('---')