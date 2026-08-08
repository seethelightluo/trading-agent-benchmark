p='scripts/miner_1_20300613_lagged_downside_close_location_asymmetry_60obs.py'
s=open(p).read();s=s.replace(" if isinstance(cond,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a]) for a in x.columns})"," if isinstance(x,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a] if isinstance(cond,pd.DataFrame) else cond) for a in x.columns})")
open(p,'w').write(s)
