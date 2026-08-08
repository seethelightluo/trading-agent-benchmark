"""Fast validation of one continuous defensive-metal transmission candidate; no persistence eligibility without full library test."""
from pathlib import Path
s=Path('scripts/miner_2_20330929_fast_defensive_metal_relative_shock_30.py').read_text()
s=s.replace("END=pd.Timestamp('2033-09-28')", "END=pd.Timestamp('2033-10-26')")
# Replace binary conditional driver construction with continuous standardized relative-metal shock.
start="""def conditional_beta(mask):
 x=S.where(mask)
 den=x.rolling(30,min_periods=8).var()
 return pd.DataFrame({a:R[a].where(mask).rolling(30,min_periods=8).cov(x).div(den) for a in A})
raw=-(conditional_beta(M<0)-conditional_beta(M>=0))"""
rep="""driver=(S-S.rolling(60,min_periods=40).mean())/(S.rolling(60,min_periods=40).std()+1e-12)
den=driver.rolling(30,min_periods=15).var()
raw=pd.DataFrame({a:-R[a].rolling(30,min_periods=15).cov(driver).div(den) for a in A})"""
assert start in s
s=s.replace(start,rep).replace('inverse_defensive_metal_relative_shock_transmission_residual_30','continuous_inverse_defensive_metal_relative_shock_transmission_residual_30')
exec(compile(s,'fast_continuous_defensive_metal_20331027','exec'))
