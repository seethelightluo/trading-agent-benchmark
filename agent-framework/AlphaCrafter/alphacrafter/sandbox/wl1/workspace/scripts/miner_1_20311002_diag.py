exec(open('scripts/miner_3_20311002_adaptive_recovery.py').read().replace("rows=[]","print(p.shape,p.index.min(),p.index.max(),m60.notna().sum().sum(),down40.notna().sum().sum()); rows=[]"))
