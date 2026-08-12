# Patch zero-volatility artifacts before validation; otherwise same candidate.
p='scripts/miner_3_20320401_residual_breadth.py'
s=open(p).read();s=s.replace("f=(-res.div(v)).where(breadth<.60,np.nan)","f=(-res.div(v.replace(0,np.nan))).where(breadth<.60,np.nan).replace([np.inf,-np.inf],np.nan)")
open(p,'w').write(s)
PY
python scripts/miner_3_20320401_residual_breadth.py 2>/dev/null | head -n 20
PY