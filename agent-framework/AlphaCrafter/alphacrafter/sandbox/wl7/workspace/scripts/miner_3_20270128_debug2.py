exec(open('scripts/miner_3_20270128_debug.py').read().split("print('assets'")[0])
for s in C: print(s,C[s].first_valid_index(),C[s].last_valid_index(),C[s].notna().sum())
print('r20',r20.notna().sum());print('v',v.notna().sum());print('F',F.notna().sum())
