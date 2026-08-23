import os, json, glob, time
active = [f for f in os.listdir('factors/') if os.path.isfile(os.path.join('factors', f)) and f.endswith('.json') and not f.endswith('.bak')]
print("active library files:", active)

recent = sorted(glob.glob('scripts/*.json'), key=os.path.getmtime)
print("\n5 most recently modified scripts/*.json:")
for p in recent[-5:]:
    print(os.path.basename(p), time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))))

with open('factors/evicted/mom_10d_skip5.json.reason.json') as f:
    print("\nmom_10d_skip5 eviction reason:", json.load(f))

with open('strategy.py') as f:
    content = f.read()
print("\nstrategy.py length:", len(content))