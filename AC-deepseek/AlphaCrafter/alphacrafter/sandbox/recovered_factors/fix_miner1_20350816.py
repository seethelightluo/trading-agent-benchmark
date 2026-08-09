p='scripts/miner_1_20350816_residual_vix_shock_conditional_resilience_60_20.py'
s=open(p).read().replace("int(sig.notna().sum())","int(sig.notna().sum().sum())")
open(p,'w').write(s)
