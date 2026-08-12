# replace activation section with diagnostic-compatible direct masking
p='scripts/miner_3_20321028_vix_breadth_capitulation.py'
s=open(p).read();s=s.replace("f=(-shock/vol.replace(0,np.nan)).mul(active.replace(0,np.nan),axis=0)","base=(-shock/vol.replace(0,np.nan)); f=base.where(active.astype(bool),np.nan); print('active_rows',int(active.sum()),'finite',int(np.isfinite(f.to_numpy()).sum()))")
open(p,'w').write(s)
