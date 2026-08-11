import inspect
import alphacrafter.sim.utils.rebalance_to_weights as m
src = inspect.getsource(m)
import re
# find the definition of gross_edge_bps
idx = src.find('def gross_edge_bps')
if idx >= 0:
    print(src[idx:idx+1500])
else:
    # search imports
    for line in src.splitlines():
        if 'gross_edge' in line or 'from' in line or 'import' in line:
            print(line)
