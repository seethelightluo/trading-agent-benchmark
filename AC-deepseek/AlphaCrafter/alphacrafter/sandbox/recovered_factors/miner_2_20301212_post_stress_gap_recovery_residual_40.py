"""Miner_2 single-idea validation: post-stress intraday gap-recovery residual 40."""
from pathlib import Path
src=Path('scripts/miner_2_20301031_post_stress_recovery_breadth_residual_40.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-30')", "E=pd.Timestamp('2030-12-11')")
src=src.replace('post_stress_recovery_breadth_residual_40','post_stress_gap_recovery_residual_40')
old="""# Breadth of asset-specific relative wins immediately after broad one-day stress, net of its ordinary win frequency.
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20)).shift(1)
relative_win=R.gt(R.median(axis=1),axis=0).astype(float)
post_win=relative_win.where(stressmask,axis=0).rolling(40,min_periods=6).mean()
ordinary_win=relative_win.rolling(40,min_periods=25).mean()
F=res(post_win-ordinary_win,v,peer,dba,trend)"""
new="""# Candidate: after an extreme cross-asset down session, measure each asset's next-session
# intraday recovery from its opening gap, versus its ordinary gap recovery. This captures
# revealed demand after stress without requiring volume. The stress flag is lagged, so
# all contemporaneous OHLC observations are completed before a next-day decision.
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A})
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20)).shift(1)
gap=O/P.shift(1)-1
recovery=(P/O-1)/(gap.abs()+v+1e-12)
post=recovery.where(stressmask,axis=0).rolling(40,min_periods=6).mean()
ordinary=recovery.rolling(40,min_periods=25).mean()
F=res(post-ordinary,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('anchor absent')
src=src.replace(old,new)
exec(compile(src,'post_stress_gap_recovery_20301212','exec'))
