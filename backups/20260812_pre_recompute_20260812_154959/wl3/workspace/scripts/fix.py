p='scripts/miner_2_20260716_range_pressure_reversal.py'
s=open(p).read().replace("x=pd.concat(frames).dropna();", "x=pd.concat(frames,ignore_index=True).dropna();").replace("z=pd.concat(z).dropna();", "z=pd.concat(z,ignore_index=True).dropna();")
open(p,'w').write(s)
