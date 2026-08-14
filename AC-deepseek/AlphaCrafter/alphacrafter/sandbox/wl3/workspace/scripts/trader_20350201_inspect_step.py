import json

with open('../logs/trader_agent.json') as f:
    data = json.load(f)

print("total entries:", len(data))

# Search for step tool call results in the last 10 entries
for entry in data[-12:]:
    out = entry.get('output')
    if isinstance(out, list):
        for o in out:
            s = str(o)
            if ('Period Return' in s or 'Advanced' in s or 'rebalance' in s.lower()
                    or 'gross edge' in s.lower() or 'executed' in s.lower()):
                print('--- entry', entry.get('iteration'), entry.get('event'), '---')
                print(s[:3000])
                print()
