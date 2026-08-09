"""Miner_2 single-idea validation: market-down-session intraday recovery residual 30."""
from pathlib import Path
src=Path('scripts/miner_2_20301031_post_stress_recovery_breadth_residual_40.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-30')", "E=pd.Timestamp('2031-01-22')")
src=src.replace('post_stress_recovery_breadth_residual_40','market_down_session_intraday_recovery_residual_30')
old="""# Breadth of asset-specific relative wins immediately after broad one-day stress, net of its ordinary win frequency.
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20)).shift(1)
relative_win=R.gt(R.median(axis=1),axis=0).astype(float)
post_win=relative_win.where(stressmask,axis=0).rolling(40,min_periods=6).mean()
ordinary_win=relative_win.rolling(40,min_periods=25).mean()
F=res(post_win-ordinary_win,v,peer,dba,trend)"""
new="""# Candidate: on completed broad-market-down sessions, measure each asset's
# open-to-close directional efficiency (close-open scaled by its daily range), less
# ordinary efficiency. It targets instruments that recover intraday under broad
# pressure rather than merely displaying a high close location; standard controls
# remove volatility, crowding, asymmetric beta and trend.
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A})
eff=(P-O)/(H-Lo).replace(0,np.nan)
market_down=M.lt(0).shift(1)
down_eff=eff.where(market_down,axis=0).rolling(30,min_periods=8).mean()
ordinary=eff.rolling(30,min_periods=20).mean()
F=res(down_eff-ordinary,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new)
exec(compile(src,'market_down_session_intraday_recovery_20310123','exec'))
