"""Scheduled revalidation of residual return-autocorrelation expansion (20d vs 60d).
Uses only completed daily observations through 2031-09-17."""
import pathlib
src=pathlib.Path('scripts/miner_1_20310724_revalidate_residual_return_autocorrelation_expansion_20_60d.py').read_text(encoding='utf8')
src=src.replace("END=pd.Timestamp('2031-07-23')", "END=pd.Timestamp('2031-09-17')")
exec(compile(src, 'scripts/miner_1_20310918_revalidate_residual_return_autocorrelation_expansion_20_60d.py', 'exec'))
