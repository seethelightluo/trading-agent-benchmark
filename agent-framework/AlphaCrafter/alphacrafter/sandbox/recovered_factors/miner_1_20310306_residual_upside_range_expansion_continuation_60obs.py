"""Single idea: residual upside range-expansion continuation, 60 observations.
Higher signal means an asset has recently shown larger intraday ranges specifically
on prior idiosyncratic-positive days, normalized by its own typical range.
This tests whether selective upside participation predicts later relative returns."""
from pathlib import Path
src=Path('scripts/miner_1_20310220_residual_upside_close_location_resilience_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2031-02-19')", "END=pd.Timestamp('2031-03-05')")
src=src.replace('residual_upside_close_location_resilience_60obs','residual_upside_range_expansion_continuation_60obs')
src=src.replace('residual-upside close-location resilience','residual-upside range-expansion continuation')
src=src.replace('mean close location conditional on prior-day positive market-residual return','mean normalized intraday range conditional on prior-day positive market-residual return')
old="""loc0=(p-lo).div((hi-lo).replace(0,np.nan))
basebeta=pd.DataFrame({a:r[a].rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var()) for a in A})
eps=r-basebeta.mul(m,axis=0)
# Conditional mean requires at least 12 qualifying observations in the last 60.
f=pd.DataFrame({a:loc0[a].where(eps[a].shift()>0).rolling(60,min_periods=12).mean() for a in A})"""
new="""# Intraday range scaled by each asset's trailing median range removes
# cross-asset price-scale and usual-volatility differences.
rng=(hi-lo).div(p.shift()).abs()
normrng=rng.div(rng.rolling(20,min_periods=15).median().replace(0,np.nan))
basebeta=pd.DataFrame({a:r[a].rolling(60,min_periods=30).cov(m).div(m.rolling(60,min_periods=30).var()) for a in A})
eps=r-basebeta.mul(m,axis=0)
# At least 12 idiosyncratic-upside observations required.
f=pd.DataFrame({a:normrng[a].where(eps[a].shift()>0).rolling(60,min_periods=12).mean() for a in A})"""
assert old in src
src=src.replace(old,new)
src=src.replace('mean_60(close_location[t] | residual_return[t-1] > 0)','mean_60((high-low)/close/median_20(range) | residual_return[t-1] > 0)')
src=src.replace('miner_1_20310220_residual_upside_close_location_resilience_60obs_candidate_signal.pkl','miner_1_20310306_residual_upside_range_expansion_continuation_60obs_candidate_signal.pkl')
exec(compile(src,'residual_upside_range_harness','exec'))
