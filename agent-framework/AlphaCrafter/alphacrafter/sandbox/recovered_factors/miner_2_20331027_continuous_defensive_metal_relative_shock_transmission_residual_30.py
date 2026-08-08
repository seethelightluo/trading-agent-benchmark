"""Single candidate: continuous defensive-metal relative-shock transmission residual, full-library screen."""
from pathlib import Path
src=Path('scripts/miner_2_20330901_inverse_cross_asset_dispersion_shock_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2033-08-31')", "E=pd.Timestamp('2033-10-26')")
old="""# Pre-specified candidate: abnormal cross-asset dispersion, standardized against its
# trailing 60-session distribution. The inverse 30-session transmission beta identifies
# assets relatively insulated from broad disagreement/repricing, after generic risk,
# crowding, downside-beta asymmetry, and trend controls.
disp=R.std(axis=1)
driver=(disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(driver,driver.notna(),30,15),v,peer,dba,trend)"""
new="""# Pre-specified candidate: continuous defensive-metal relative shock, standardized
# against its trailing 60-session distribution. Inverse 30-session transmission beta
# identifies assets insulated from flight-to-safety (gold versus copper) repricing.
# Controls remove generic risk, peer crowding, downside-beta asymmetry, and trend.
metal=R.XAU-R.COPPER
driver=(metal-metal.rolling(60,min_periods=40).mean())/(metal.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(driver,driver.notna(),30,15),v,peer,dba,trend)"""
assert old in src
src=src.replace(old,new).replace('inverse_cross_asset_dispersion_shock_transmission_residual_30','continuous_inverse_defensive_metal_relative_shock_transmission_residual_30')
exec(compile(src,'miner_2_20331027_continuous_defensive_metal.py','exec'))
