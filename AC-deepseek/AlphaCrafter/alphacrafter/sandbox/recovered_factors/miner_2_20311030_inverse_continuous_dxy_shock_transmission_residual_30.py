"""Miner_2 single-idea validation: inverse continuous DXY shock transmission residual."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-10-29')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Pre-specified candidate: inverse 30-session beta to a continuous standardized DXY shock.
# Higher scores indicate lower dollar-shock transmission after cross-sectional removal
# of realized risk, peer crowding, downside beta asymmetry, and medium-term trend.
# DXY is observation-only and used solely as an available macro driver.
raw=rd('DXY',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
fx=raw/(raw.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(fx,fx.notna()),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", """# Later admitted signal families included in the point-in-time novelty screen.
cb=R[['WTI','COPPER','XAU']].mean(1); eb=R[['SPX','NDX','SX5E','HSI']].mean(1); div=cb-eb
L['commodity_equity_divergence_asymmetry']=res(-(beta(div,div>=0)-beta(div,div<0)),v,peer,dba,trend)
rs=R['US10Y']-R['CN10Y']; rsz=rs/(rs.rolling(60,min_periods=40).std()+1e-12)
L['inverse_continuous_rate_spread_surprise']=res(-beta(rsz,rsz.notna()),v,peer,dba,trend)
st=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)
L['inverse_equity_stress_rate_transmission']=res(-(beta(R['US10Y']*st,st.notna())-beta(R['US10Y']*(1-st.clip(upper=1)),st.notna())),v,peer,dba,trend)
print('FACTOR inverse_continuous_dxy_shock_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals_reconstructed',len(L))""")
exec(compile(src,'inverse_continuous_dxy_shock_20311030','exec'))
