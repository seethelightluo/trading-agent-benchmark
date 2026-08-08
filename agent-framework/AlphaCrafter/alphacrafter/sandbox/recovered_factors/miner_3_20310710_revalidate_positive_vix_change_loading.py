"""Scheduled 2031-07-10 revalidation: residual positive VIX-change shock loading contraction."""
import pathlib
src=pathlib.Path('scripts/miner_3_20300221_revalidate_positive_vix_change_shock_loading_contraction.py').read_text(encoding='utf8')
src=src.replace("END=pd.Timestamp('2030-02-20')", "END=pd.Timestamp('2031-07-09')")
exec(compile(src,'vix_loading_contraction_revalidation_20310710','exec'))
