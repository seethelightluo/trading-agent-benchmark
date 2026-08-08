"""Miner_2 single-idea validation: inverse cross-crypto dispersion shock transmission residual (30).
Uses daily data available through 2031-12-10; forward returns are formed only inside cutoff.
"""
from pathlib import Path
src=Path('scripts/miner_3_20311127_inverse_copper_oil_relative_shock_transmission_beta_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2031-11-26')", "E=pd.Timestamp('2031-12-10')")
old="""# Pre-specified candidate: inverse beta to a continuous copper-versus-oil relative shock.
# The driver is standardized COPPER return minus WTI return, a within-commodity signal
# separating industrial-growth demand from broad energy/inflation shocks. Higher scores
# favor assets with lower transmission to this relative-demand shock after standard controls.
raw=R['COPPER']-R['WTI']
fx=raw/(raw.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(fx,fx.notna()),v,peer,dba,trend)"""
new="""# Pre-specified candidate: inverse beta to cross-crypto dispersion shocks.
# Driver is the absolute BTC-minus-ETH daily return gap, standardized against its own
# 60-day history. It isolates disagreement within crypto from the common crypto direction.
# Higher values favor lower transmission to idiosyncratic crypto-dislocation shocks,
# residualized from volatility, peer crowding, market beta asymmetry and trend.
raw=(R['BTC']-R['ETH']).abs()
fx=(raw-raw.rolling(60,min_periods=40).mean())/(raw.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(fx,fx.notna()),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new)
src=src.replace("inverse_copper_oil_relative_shock_transmission_beta_residual_30", "inverse_cross_crypto_dispersion_shock_transmission_residual_30")
Path('scripts/miner_2_20311211_inverse_cross_crypto_dispersion_shock_transmission_residual_30.py').write_text(src)
print('wrote candidate script')
