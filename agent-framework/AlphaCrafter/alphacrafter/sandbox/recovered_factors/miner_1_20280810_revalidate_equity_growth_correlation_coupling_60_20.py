"""Miner 1 quarterly PIT revalidation, equity-growth correlation coupling, full effective library."""
import pathlib
src=pathlib.Path('scripts/miner_1_20280518_revalidate_equity_growth_correlation_coupling_60_20.py').read_text()
src=src.replace("END=pd.Timestamp('2028-05-17')", "END=pd.Timestamp('2028-08-09')")
# The historical source's library predates four now-effective factors and contains one
# retired serial signal. Amend only after all inherited signal inputs are available.
needle="print('FACTOR equity_growth_correlation_coupling_60_20 REVALIDATION_END'"
insert="""# Bring independence evidence to the current active library (candidate itself excluded).
lib.pop('miner_2_residual_downside_serial_reversal_60d', None)
def_basket=r[['XAU','US10Y','CN10Y']].mean(axis=1)
def20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(def_basket) for a in A})
def60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(def_basket) for a in A})
lib['miner_1_residualized_defensive_correlation_decoupling_60_20']=residual(-(def20-def60),(p/p.shift(20)-1)/own,own)
inf_basket=r[['COPPER','WTI']].mean(axis=1)
inf20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(inf_basket) for a in A})
inf60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(inf_basket) for a in A})
lib['miner_1_residualized_inflation_basket_correlation_decoupling_60_20']=residual(-(inf20-inf60),(p/p.shift(20)-1)/own,own)
lib['miner_1_residualized_return_autocorrelation_20d']=residual(pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(r[a].shift(1)) for a in A}),(p/p.shift(20)-1)/own,own)
svp=(np.sign(e)*lv.diff()).where(e<0)
lib['miner_3_residual_downside_signed_volume_pressure_deceleration_20_60d']=-(svp.rolling(20,min_periods=8).mean()-svp.rolling(60,min_periods=18).mean())/(e.rolling(60,min_periods=40).std()+1e-12)
"""
src=src.replace(needle,insert+"\n"+needle.replace(' REVALIDATION_END',' QUARTERLY_REVALIDATION_END'))
exec(src)
