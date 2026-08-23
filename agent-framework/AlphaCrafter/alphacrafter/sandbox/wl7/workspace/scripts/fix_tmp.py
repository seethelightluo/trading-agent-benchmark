p='scripts/miner_2_20270526_volume_surprise_reversal.py'
s=open(p).read().replace("x=pd.concat(rr).replace([np.inf,-np.inf],np.nan).dropna();", "x=pd.concat(rr,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna();")
open(p,'w').write(s)
