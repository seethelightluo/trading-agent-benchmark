import os
f=os.listdir('factors')
active=[x for x in f if x.endswith('.json') and not x.endswith('.bak') and not x.endswith('.reason.json')]
print("active json files:", active)
print("count:", len(active))