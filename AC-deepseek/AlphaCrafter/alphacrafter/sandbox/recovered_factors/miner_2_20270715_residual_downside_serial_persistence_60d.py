"""miner_2: validate one idea -- residual downside serial persistence, 60d."""
import pathlib
src=pathlib.Path('scripts/miner_3_20270422_residual_upside_market_down_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-04-22')", "END=pd.Timestamp('2027-07-15')")
src=src.replace("# Mean positive idiosyncratic return only on broad-market down sessions, normalized by residual risk.\n# High values identify assets that retain independent upside when the cross-asset tape declines.\ndown=e.where(m<0, np.nan).clip(lower=0)\nf=down.rolling(60,min_periods=12).mean()/e.rolling(60,min_periods=40).std()", "# Lag-one autocorrelation of the downside component of market-neutral residual returns.\n# Setting non-downside observations to zero preserves a common 60-day history while\n# isolating the temporal persistence of idiosyncratic losses.\nneg=e.clip(upper=0)\nf=pd.DataFrame({a: neg[a].rolling(60,min_periods=45).corr(neg[a].shift(1)) for a in A})")
src=src.replace("FACTOR residual_upside_in_market_down_60d", "FACTOR residual_downside_serial_persistence_60d")
src=src.replace("if h==5:", "if h==1:")
pathlib.Path('scripts/miner_2_20270715_residual_downside_serial_persistence_60d.py').write_text(src)
print('written candidate')
