"""One idea: downside absorption volume-confirmation acceleration; cutoff 2032-01-21."""
from pathlib import Path
src=Path('scripts/miner_1_20320108_residual_downside_absorption_quality_20_60obs.py').read_text()
src=src.replace('residual-downside absorption quality; completed-bar cutoff 2032-01-07','downside absorption volume-confirmation acceleration; completed-bar cutoff 2032-01-21')
src=src.replace("END=pd.Timestamp('2032-01-07')", "END=pd.Timestamp('2032-01-21')")
old="""# A severe idiosyncratic down session that closes high in its own intraday range
# reflects absorption rather than persistent selling. Compare severity-weighted
# absorption over 20 sessions with an asset's 60-session normal level.
clv0=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
severity=(-res/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
absorption=severity*clv0
recent=absorption.rolling(20,min_periods=15).sum()/(severity.rolling(20,min_periods=15).sum()+1e-12)
baseline=absorption.rolling(60,min_periods=45).sum()/(severity.rolling(60,min_periods=45).sum()+1e-12)
f=recent-baseline"""
new="""# Severe idiosyncratic down sessions closing high in range are more credible
# absorption when participation is above an asset's own 60-session norm. The
# signal is the recent-versus-baseline change in this volume-confirmed quality.
clv0=((p-lo)/(hi-lo).replace(0,np.nan)).clip(0,1)
severity=(-res/res.rolling(60,min_periods=45).std()).clip(lower=0,upper=4)
relvol=(vo/vo.rolling(60,min_periods=45).median()).clip(0.25,4)
quality=severity*clv0*relvol
recent=quality.rolling(20,min_periods=15).sum()/(severity.rolling(20,min_periods=15).sum()+1e-12)
baseline=quality.rolling(60,min_periods=45).sum()/(severity.rolling(60,min_periods=45).sum()+1e-12)
f=recent-baseline"""
assert old in src
src=src.replace(old,new).replace('residual_downside_absorption_quality_20_60obs','downside_absorption_volume_confirmation_acceleration_20_60obs')
# include the immediately preceding admitted absorption signal in the novelty screen
needle="'inverse_dispersion_resid_persistence':latest}"
replacement="'inverse_dispersion_resid_persistence':latest,'residual_downside_absorption_quality':(severity*clv).rolling(20,min_periods=15).sum()/(severity.rolling(20,min_periods=15).sum()+1e-12)-(severity*clv).rolling(60,min_periods=45).sum()/(severity.rolling(60,min_periods=45).sum()+1e-12)}"
assert needle in src
src=src.replace(needle,replacement)
exec(compile(src,'downside_absorption_volume_confirmation_acceleration_20_60obs.py','exec'))
