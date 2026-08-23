import os
# Confirm no active (non-bak) factor serialization files exist in the admitted library
active = [f for f in os.listdir('factors/') if os.path.isfile(os.path.join('factors', f)) and f.endswith('.json') and not f.endswith('.bak')]
print("active library files:", active)

# Check if any recent miner outputs exist (empty submissions this cycle)
import glob, time
recent = sorted(glob.glob('scripts/*.json'), key=os.path.getmtime)
print("\n5 most recently modified scripts/*.json:")
for p in recent[-5:]:
    print(os.path.basename(p), time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p))))

# Check evicted reasons for one factor to confirm eviction state
with open('factors/evicted/mom_10d_skip5.json.reason.json') as f:
    print("\nmom_10d_skip5 eviction reason:", json.load(f))

# Check strategy.py operative ensemble
with open('strategy.py') as f:
    content = f.read()
import re
m = re.findall(r'factor_id["\s:]+([a-zA-Z0-9_]+)', content)
print("\nstrategy.py factor refs (sample):", m[:10])