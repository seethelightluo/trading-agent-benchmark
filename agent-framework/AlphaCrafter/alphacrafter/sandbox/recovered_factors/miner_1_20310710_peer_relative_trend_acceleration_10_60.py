"""One idea: peer-relative trend acceleration (10 versus 60 sessions), with full admitted-library novelty audit."""
# Generated from the established full-library audit framework; candidate is intentionally the only changed construction.
import pathlib
src=pathlib.Path('scripts/miner_3_20310612_peer_upside_minus_downside_correlation_asymmetry_40.py').read_text()
src=src.replace('"""One idea: peer downside-versus-upside correlation asymmetry (40), with admitted-library novelty audit."""','"""One idea: peer-relative trend acceleration (10 versus 60 sessions), with admitted-library novelty audit."""')
old="cand=cs(pd.DataFrame({a:r[a].where(m>0).rolling(40,min_periods=12).corr(m.where(m>0))-r[a].where(m<0).rolling(40,min_periods=12).corr(m.where(m<0)) for a in A})).shift(1)"
new="""# Acceleration in each asset's cross-asset-relative trend: recent 10-session relative return minus its preceding 50-session relative return, volatility-normalized.\n# This distinguishes fresh leadership from a level-only momentum signal.\nrel10=rel.rolling(10,min_periods=8).sum()\nrel60=rel.rolling(60,min_periods=45).sum()\nrecent_vol=rel.rolling(20,min_periods=15).std()*np.sqrt(10)\ncand=cs((rel10-(rel60-rel10)*.2).div(recent_vol.replace(0,np.nan))).shift(1)"""
assert old in src
src=src.replace(old,new).replace("print('FACTOR peer_upside_minus_downside_correlation_asymmetry_40 CUTOFF'", "print('FACTOR peer_relative_trend_acceleration_10_60 CUTOFF'")
# The source's event construct repaired below is valid; audit all live JSON factor definitions operationally.
exec(compile(src,'miner_1_20310710_peer_relative_trend_acceleration_10_60.py','exec'))
