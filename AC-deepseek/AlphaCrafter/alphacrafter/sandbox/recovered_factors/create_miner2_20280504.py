"""miner_2 single-idea validation: broad-drawdown residual dispersion asymmetry."""
import pathlib
src=pathlib.Path('scripts/miner_2_20280420_residual_broad_drawdown_outcome_dispersion_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2028-04-19')", "END=pd.Timestamp('2028-05-03')")
old="""# On broad-stress sessions (<=40% assets up), measure dispersion of each asset's
# idiosyncratic outcomes.  Negative orientation rewards stable residual behavior
# under common drawdown rather than the conditional residual mean already admitted.
breadth=(r>0).mean(axis=1); stress=(breadth<=.40)
es=e.where(stress, float('nan'))
f=-es.rolling(60,min_periods=20).std()/(e.rolling(60,min_periods=40).std()+1e-12)"""
new="""# Compare idiosyncratic dispersion in broad drawdowns with dispersion in broad
# advances.  High values identify assets whose residual outcomes become relatively
# more stable exactly when the cross-asset tape is stressed (a conditional
# asymmetry, rather than the level of stress-session dispersion).
breadth=(r>0).mean(axis=1); stress=(breadth<=.40); advance=(breadth>=.60)
es=e.where(stress, float('nan')); ea=e.where(advance, float('nan'))
down_sd=es.rolling(60,min_periods=20).std(); up_sd=ea.rolling(60,min_periods=12).std()
f=-(down_sd-up_sd)/(e.rolling(60,min_periods=40).std()+1e-12)"""
assert old in src
src=src.replace(old,new).replace('residual_broad_drawdown_outcome_dispersion_60d','residual_broad_drawdown_dispersion_asymmetry_60d')
pathlib.Path('scripts/miner_2_20280504_residual_broad_drawdown_dispersion_asymmetry_60d.py').write_text(src)
