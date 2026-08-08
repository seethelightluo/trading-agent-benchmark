"""Miner_2 single-idea validation: broad-market-down relative strength residual 30."""
from pathlib import Path
src=Path('scripts/miner_2_20301031_post_stress_recovery_breadth_residual_40.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-30')", "E=pd.Timestamp('2031-01-08')")
src=src.replace('post_stress_recovery_breadth_residual_40','market_down_relative_strength_residual_30')
old="""# Breadth of asset-specific relative wins immediately after broad one-day stress, net of its ordinary win frequency.
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20)).shift(1)
relative_win=R.gt(R.median(axis=1),axis=0).astype(float)
post_win=relative_win.where(stressmask,axis=0).rolling(40,min_periods=6).mean()
ordinary_win=relative_win.rolling(40,min_periods=25).mean()
F=res(post_win-ordinary_win,v,peer,dba,trend)"""
new="""# Candidate: each asset's completed-session return on broad market-down days,
# less its ordinary 30-day return. This is unconditional enough for broad coverage,
# yet isolates defensive relative strength in adverse cross-asset sessions. Controls
# remove ordinary volatility, crowding, downside-beta asymmetry and trend.
market_down=M.lt(0).shift(1)
down_mean=R.where(market_down,axis=0).rolling(30,min_periods=8).mean()
ordinary=R.rolling(30,min_periods=20).mean()
F=res((down_mean-ordinary)/(v+1e-12),v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new)
exec(compile(src,'market_down_relative_strength_20310109','exec'))
