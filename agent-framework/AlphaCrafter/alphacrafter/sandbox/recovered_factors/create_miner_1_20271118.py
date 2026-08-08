"""miner_1: validate one idea -- trend-residualized return autocorrelation."""
import pathlib
src=pathlib.Path('scripts/miner_1_20271104_residualized_realized_return_skewness_20d.py').read_text()
src=src.replace('"""miner_1: validate one factor idea -- trend-residualized 20-session return skewness."""', '"""miner_1: validate one factor idea -- trend-residualized return autocorrelation."""')
src=src.replace("END=pd.Timestamp('2027-11-03')", "END=pd.Timestamp('2027-11-17')")
old="""# Third standardized central moment of trailing returns, then daily cross-sectional
# residualization against 20d risk-adjusted trend. This targets path asymmetry
# orthogonal to ordinary medium-term momentum.
mu=r.rolling(20,min_periods=15).mean(); sig=r.rolling(20,min_periods=15).std()
raw=((r-mu)**3).rolling(20,min_periods=15).mean()/(sig**3)
trend=(p/p.shift(20)-1)/own
f=residual(raw,trend,own)"""
new="""# Lag-one autocorrelation over a trailing 20-session return path, residualized
# each day against risk-adjusted trend.  It measures persistence/choppiness not
# explained by an asset's level of medium-term trend.
def ac1(x):
    x=x.dropna()
    return x.autocorr(lag=1) if len(x)>=15 else np.nan
raw=r.rolling(20,min_periods=15).apply(ac1,raw=False)
trend=(p/p.shift(20)-1)/own
f=residual(raw,trend,own)"""
assert old in src
src=src.replace(old,new).replace('residualized_realized_return_skewness_20d','residualized_return_autocorrelation_20d')
pathlib.Path('scripts/miner_1_20271118_residualized_return_autocorrelation_20d.py').write_text(src)
print('written')
