"""miner_2: one idea -- return-volume coupling improvement, 60d versus 20d."""
import pathlib
src=pathlib.Path('scripts/miner_3_20270422_residual_upside_market_down_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-04-22')", "END=pd.Timestamp('2027-05-19')")
# Replace only the candidate construction that appears after the library has been built.
old="""# Mean positive idiosyncratic return only on broad-market down sessions, normalized by residual risk.
# High values identify assets that retain independent upside when the cross-asset tape declines.
down=e.where(m<0, np.nan).clip(lower=0)
f=down.rolling(60,min_periods=12).mean()/e.rolling(60,min_periods=40).std()"""
new="""# Change in own return--log-volume correlation: high values mean price moves have
# become more volume-confirmed recently relative to the preceding medium-term state.
# This is a cross-asset transition/participation signal, distinct from volume level.
lv=np.log(vol.replace(0,np.nan))
c60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(lv[a].pct_change()) for a in A})
c20=pd.DataFrame({a:r[a].rolling(20,min_periods=14).corr(lv[a].pct_change()) for a in A})
f=c20-c60"""
assert old in src
src=src.replace(old,new)
src=src.replace("FACTOR residual_upside_in_market_down_60d", "FACTOR return_volume_coupling_improvement_60_20")
src=src.replace("CANDIDATE", "CANDIDATE")
pathlib.Path('scripts/miner_2_20270520_return_volume_coupling_improvement_60_20.py').write_text(src)
print('wrote candidate script')
