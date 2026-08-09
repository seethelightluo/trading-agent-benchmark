"""Miner_2 single-idea exploration: inverse dispersion-amplified VIX-shock transmission residual.
Uses the established point-in-time cross-asset validation harness with only the
candidate macro driver changed from DXY to observation-only VIX."""
from pathlib import Path
src=Path('scripts/miner_3_20320930_inverse_dispersion_amplified_dxy_shock_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2032-09-29')", "E=pd.Timestamp('2032-10-27')")
src=src.replace('inverse_dispersion_amplified_dxy_shock_transmission_residual_30','inverse_dispersion_amplified_vix_shock_transmission_residual_30')
# Change solely the candidate driver. The independent library proxy block remains
# intact, including the VIX-beta proxy, so novelty is tested against VIX exposure.
src=src.replace("dxy=rd('DXY',root='../persistent/index_data/').reindex(P.index).pct_change(fill_method=None)\ng=dxy*(1+dz.clip(0,3)); F=res(-beta(g,pd.Series(True,index=P.index)),v,peer,dba,trend)", "dxy=rd('VIX',root='../persistent/index_data/').reindex(P.index).pct_change(fill_method=None)\ng=dxy*(1+dz.clip(0,3)); F=res(-beta(g,pd.Series(True,index=P.index)),v,peer,dba,trend)")
exec(compile(src, __file__, 'exec'))
