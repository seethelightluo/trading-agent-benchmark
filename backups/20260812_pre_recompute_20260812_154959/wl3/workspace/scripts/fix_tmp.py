p='scripts/miner_2_20300418_dxy_residual_reversal.py'
s=open(p).read(); s=s.replace("y=r.shift(-1).rolling(h).sum() if h>1 else r.shift(-1)","y=sum(r.shift(-k) for k in range(1,h+1))")
open(p,'w').write(s)
