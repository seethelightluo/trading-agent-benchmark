p='scripts/miner_1_20290906_vix_residual.py'
s=open(p).read().replace("stress=max(0.,min(2.,float(v.iloc[t]/vm-1)*5+1))", "stress=2.0 if v.iloc[t]>vm else 0.0")
s=s.replace("miner_1_20290906_vix_residual_signal.csv", "miner_1_20290906_vix_binary_residual_signal.csv")
open('scripts/miner_1_20290906_vix_binary_residual.py','w').write(s)
