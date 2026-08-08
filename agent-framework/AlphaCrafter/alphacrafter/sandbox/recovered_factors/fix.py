p='scripts/miner_2_20280727_revalidate_commonality_expansion_transition_40.py'
s=open(p).read()
s=s.replace(" z=np.array(z); sd=z.std(ddof=1) if len(z)>1 else np.nan\n return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/sd),6),hit=round(float((z>0).mean()),6),mean_n=round(float(np.mean(ns)),3),min_n=int(min(ns)))", " z=np.array(z)\n if not len(z): return dict(dates=0,ic=None,icir=None,hit=None,mean_n=None,min_n=None)\n sd=z.std(ddof=1) if len(z)>1 else np.nan\n return dict(dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/sd),6),hit=round(float((z>0).mean()),6),mean_n=round(float(np.mean(ns)),3),min_n=int(min(ns)))")
open(p,'w').write(s)
