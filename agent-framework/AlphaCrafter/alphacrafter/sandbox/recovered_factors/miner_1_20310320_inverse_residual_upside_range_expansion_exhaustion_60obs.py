"""Single idea: inverse residual-upside range expansion exhaustion, 60 observations.
The factor is the negative of normalized intraday range averaged only when the
prior asset-specific market-residual return was positive. A high score therefore
identifies assets without repeated abnormal upside-state range expansion, testing
whether the prior adverse range-expansion relation is a cross-asset exhaustion signal."""
from pathlib import Path
src=Path('scripts/miner_1_20310306_residual_upside_range_expansion_continuation_60obs.py').read_text()
src=src.replace("END=pd.Timestamp('2031-03-05')", "END=pd.Timestamp('2031-03-19')")
src=src.replace('residual_upside_range_expansion_continuation_60obs','inverse_residual_upside_range_expansion_exhaustion_60obs')
src=src.replace('residual-upside range-expansion continuation','inverse residual-upside range-expansion exhaustion')
src=src.replace('Higher signal means an asset has recently shown larger intraday ranges specifically\non prior idiosyncratic-positive days, normalized by its own typical range.\nThis tests whether selective upside participation predicts later relative returns.', 'Higher signal means less repeated abnormal intraday range expansion on prior idiosyncratic-positive days. This tests the explicitly inverted exhaustion relation.')
src=src.replace("f=pd.DataFrame({a:normrng[a].where(eps[a].shift()>0).rolling(60,min_periods=12).mean() for a in A})", "f=pd.DataFrame({a:-normrng[a].where(eps[a].shift()>0).rolling(60,min_periods=12).mean() for a in A})")
src=src.replace('mean_60((high-low)/close/median_20(range) | residual_return[t-1] > 0)', '-mean_60((high-low)/prior_close/median_20(range) | residual_return[t-1] > 0)')
src=src.replace('miner_1_20310306_residual_upside_range_expansion_continuation_60obs_candidate_signal.pkl','miner_1_20310320_inverse_residual_upside_range_expansion_exhaustion_60obs_candidate_signal.pkl')
exec(compile(src,'inverse_residual_upside_range_harness','exec'))
