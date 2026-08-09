"""One idea: cumulative residual recovery efficiency conditional on severe idiosyncratic losses."""
from pathlib import Path
src=Path('scripts/miner_1_20311211_severity_weighted_immediate_residual_rebound_quality_10_60obs.py').read_text()
src=src.replace('severity-weighted immediate residual rebound quality; 2031-12-10 completed-bar cutoff','cumulative residual recovery efficiency conditional on severe idiosyncratic losses; 2031-12-24 completed-bar cutoff')
src=src.replace("END=pd.Timestamp('2031-12-10')", "END=pd.Timestamp('2031-12-24')")
old="""# An immediate idiosyncratic rebound earns a high score only when it follows a severe
# prior-day idiosyncratic loss.  The 10-observation severity-weighted mean is divided
# by its 60-observation counterpart, isolating recovery quality from an asset's level
# of typical residual volatility.
prior_severity=(-res.shift(1)/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
immediate_rebound=res*prior_severity
recent=immediate_rebound.rolling(10,min_periods=7).mean()
baseline=immediate_rebound.rolling(60,min_periods=45).mean()
f=recent-baseline"""
new="""# Recovery efficiency: after a severe idiosyncratic down day, count only the
# positive part of the next-day residual response, scaled by loss severity.  A 10d
# efficiency minus its 60d normal level measures whether downside events are being
# repaired unusually cleanly rather than merely followed by noisy one-day bounces.
prior_severity=(-res.shift(1)/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
recovery=res.clip(lower=0)*prior_severity
loss_mass=prior_severity
recent=recovery.rolling(10,min_periods=7).sum()/(loss_mass.rolling(10,min_periods=7).sum()+1e-12)
baseline=recovery.rolling(60,min_periods=45).sum()/(loss_mass.rolling(60,min_periods=45).sum()+1e-12)
f=recent-baseline"""
assert old in src
src=src.replace(old,new).replace('severity_weighted_immediate_residual_rebound_quality_10_60obs','conditional_residual_recovery_efficiency_10_60obs')
exec(compile(src,'conditional_residual_recovery_efficiency_10_60obs.py','exec'))
