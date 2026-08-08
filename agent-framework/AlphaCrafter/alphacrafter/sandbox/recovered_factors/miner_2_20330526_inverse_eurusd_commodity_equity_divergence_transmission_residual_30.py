"""Miner_2 one-idea validation: inverse EURUSD transmission beta conditional on commodity/equity divergence.
Visible-data cutoff 2033-05-25.  The signal is a distinct conditional macro channel, residualized
from generic risk and trend; all forward returns are formed only after each historical signal date.
"""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2033-05-25')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: inverse beta to standardized EURUSD shocks when 5-day commodity and equity baskets diverge.
# The divergence mask has no directional overlap with simple FX shock-asymmetry: it requires
# opposite 5-day moves of the commodity (XAU,COPPER,WTI) and equity-index baskets.
# Residualization removes own volatility, peer crowding, market downside beta asymmetry and trend.
eur=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
eur=eur/(eur.rolling(60,min_periods=40).std()+1e-12)
cmd=R[['XAU','COPPER','WTI']].mean(axis=1).rolling(5,min_periods=4).sum()
eq=R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1).rolling(5,min_periods=4).sum()
div=(cmd*eq<0) & ((cmd-eq).abs() > (cmd-eq).abs().rolling(60,min_periods=40).median())
F=-res(beta(eur,div,30,10),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("print('FACTOR oil_market_drawdown_conditional_transmission_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR inverse_eurusd_commodity_equity_divergence_transmission_residual_30 EXPLORATION visible_through',E.date(),'assets',len(A),'library_proxy_signals',len(L))")
# clarify this historical harness library is proxy-only; it remains sufficient to reject but cannot admit without full evidence
src=src.replace("# Complete reconstructed admitted-library screen", "# Historical reconstructed library proxy screen (a candidate cannot be admitted from proxy evidence alone).")
exec(compile(src,'miner_2_eurusd_divergence_20330526','exec'))
