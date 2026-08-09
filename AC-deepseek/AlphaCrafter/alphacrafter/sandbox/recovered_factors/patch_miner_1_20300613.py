# Patch the research script's beta helper to align macro series to each asset's native observation index before rolling conditional covariance.
p='scripts/miner_1_20300613_lagged_downside_close_location_asymmetry_60obs.py'
s=open(p).read()
old="""def beta(x,y,w=60,cond=None):
 if isinstance(cond,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a]) for a in x.columns})
 yy=y.where(cond) if cond is not None else y; xx=x.where(cond) if cond is not None else x
 return xx.rolling(w,min_periods=12).cov(yy).div(yy.rolling(w,min_periods=12).var())"""
new="""def beta(x,y,w=60,cond=None):
 if isinstance(cond,pd.DataFrame): return pd.DataFrame({a:beta(x[a],y,w,cond[a]) for a in x.columns})
 # Macro observations are forward-filled only onto an asset's completed native-bar dates;
 # no later macro value is used. This avoids union-calendar rolling-window emptiness.
 yy=y.reindex(x.index,method='ffill')
 cc=cond.reindex(x.index).fillna(False) if cond is not None else None
 xx=x.where(cc) if cc is not None else x; yy=yy.where(cc) if cc is not None else yy
 return xx.rolling(w,min_periods=12).cov(yy).div(yy.rolling(w,min_periods=12).var())"""
assert old in s
open(p,'w').write(s.replace(old,new))
