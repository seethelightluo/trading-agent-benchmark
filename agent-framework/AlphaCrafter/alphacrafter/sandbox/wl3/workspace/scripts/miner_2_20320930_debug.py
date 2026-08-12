# debug existing research: inspect alignment and valid counts
exec(open('scripts/miner_2_20320930_panic_reversal.py').read().split("ics=[]")[0])
print(p.notna().sum().to_dict()); print(sig.notna().sum().head()); print(fwd.notna().sum().head()); print(pd.concat([sig.iloc[-100],fwd.iloc[-100]],axis=1).dropna())
