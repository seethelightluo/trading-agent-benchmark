"""Miner_2 single-idea validation: continuous market-stress close-location resilience residual 30."""
from pathlib import Path
src=Path('scripts/miner_2_20301031_post_stress_recovery_breadth_residual_40.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-30')", "E=pd.Timestamp('2030-12-25')")
src=src.replace('post_stress_recovery_breadth_residual_40','continuous_stress_close_location_resilience_residual_30')
old="""# Breadth of asset-specific relative wins immediately after broad one-day stress, net of its ordinary win frequency.
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20)).shift(1)
relative_win=R.gt(R.median(axis=1),axis=0).astype(float)
post_win=relative_win.where(stressmask,axis=0).rolling(40,min_periods=6).mean()
ordinary_win=relative_win.rolling(40,min_periods=25).mean()
F=res(post_win-ordinary_win,v,peer,dba,trend)"""
new="""# Candidate: continuous, lagged broad-market downside severity weights each asset's
# completed-session close location. It measures whether an asset tends to finish near
# its high when the preceding market session was stressed, less its ordinary close
# location. Residual controls remove volatility, crowding, beta asymmetry and trend.
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A})
location=(P-Lo)/(H-Lo).replace(0,np.nan)
stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=40).std()+1e-12)).clip(0,4)
weighted=location.mul(stress,axis=0).rolling(30,min_periods=12).sum().div(stress.rolling(30,min_periods=12).sum().replace(0,np.nan),axis=0)
ordinary=location.rolling(30,min_periods=20).mean()
F=res(weighted-ordinary,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('factor anchor absent')
src=src.replace(old,new)
exec(compile(src,'continuous_stress_close_location_20301226','exec'))
