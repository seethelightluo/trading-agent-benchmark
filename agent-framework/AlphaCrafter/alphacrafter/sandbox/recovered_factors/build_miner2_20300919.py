from pathlib import Path
p=Path('scripts/miner_3_20300905_inverse_us10y_magnitude_tier_shock_transmission_residual_30.py')
s=p.read_text()
s=s.replace('"""Miner_3 candidate: inverse US10Y high-magnitude versus normal-magnitude shock transmission residual."""','"""Miner_2 candidate: tail-correlation-asymmetry acceleration residual; point-in-time research."""')
s=s.replace("A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2030-09-04')", "A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2030-09-18')")
old="fx=R.US10Y; mag=fx.abs(); threshold=mag.rolling(60,min_periods=40).median(); F=res(-(beta(fx,mag>=threshold)-beta(fx,mag<threshold)),v,peer,dba,trend)"
new="""# Candidate: improvement/deterioration in incremental correlation on the worst 20% cross-asset-market days.
tailmask=M.le(M.rolling(60,min_periods=40).quantile(.20))
tailcorr=pd.DataFrame({a:R[a].where(tailmask).rolling(60,min_periods=8).corr(M.where(tailmask)) for a in A})
allcorr=pd.DataFrame({a:R[a].rolling(60,min_periods=40).corr(M) for a in A})
tailbase=-(tailcorr-allcorr)
F=res(tailbase-tailbase.shift(20),v,peer,dba,trend)"""
assert old in s
s=s.replace(old,new)
s=s.replace("'market_down_range_expansion_recovery'=res(rng.where(M<0).rolling(20,min_periods=6).mean()/(v+1e-12),mom,peer)", "'market_down_range_expansion_recovery'=res(rng.where(M<0).rolling(20,min_periods=6).mean()/(v+1e-12),mom,peer)\nL['tail_correlation_asymmetry_residual_60']=res(tailbase,v,peer,dba,trend)")
s=s.replace('FACTOR inverse_us10y_magnitude_tier_shock_transmission_residual_30','FACTOR tail_correlation_asymmetry_acceleration_residual_60')
p=Path('scripts/miner_2_20300919_tail_correlation_asymmetry_acceleration_residual_60.py');p.write_text(s)
print(p)
