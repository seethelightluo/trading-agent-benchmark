# quick diagnostics
exec(open('scripts/miner_2_20350202_recovery_downside.py').read().split("low=p.shift(1)")[0])
print(p.notna().sum().to_dict()); print(r.notna().sum().to_dict())
