"""Miner_2 single-idea exploration: downside overnight-gap fill resilience residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-09-17')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: downside overnight-gap fill resilience. For each asset, measure
# normalized close-versus-open return only on negative open-to-prior-close gaps,
# averaged over 20 sessions. Higher values identify assets which absorb adverse
# overnight repricing. Residualize generic volatility, peer crowding, downside
# beta asymmetry and trend.
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A})
gap=O/P.shift(1)-1
intr=(P-O)/(H-Lo).replace(0,np.nan)
raw=pd.DataFrame({a:intr[a].where(gap[a]<0).rolling(20,min_periods=5).mean() for a in A})
F=res(raw,v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','downside_overnight_gap_fill_resilience_residual_20')
# In this data snapshot the inherited IC indices are session labels, so avoid invalid timestamp comparisons.
src=src.replace("for n,m in [('2026_27',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_current',ics[20].index>='2028-01-01')]:", "for n,m in []:")
exec(compile(src,'miner_2_downside_overnight_gap_fill_20310918','exec'))
