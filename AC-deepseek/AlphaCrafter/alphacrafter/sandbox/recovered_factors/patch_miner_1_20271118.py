import pathlib
p=pathlib.Path('scripts/miner_1_20271118_residualized_return_autocorrelation_20d.py')
s=p.read_text()
needle="lib['miner_3_realized_volatility_compression_20_60d']=comp\n"
insert="""lib['miner_3_realized_volatility_compression_20_60d']=comp
# Latest admitted skewness factor, included as required independence evidence.
mu=r.rolling(20,min_periods=15).mean(); sig=r.rolling(20,min_periods=15).std()
skew=((r-mu)**3).rolling(20,min_periods=15).mean()/(sig**3)
lib['miner_1_residualized_realized_return_skewness_20d']=residual(skew,trend,own)
"""
assert needle in s
p.write_text(s.replace(needle,insert))
