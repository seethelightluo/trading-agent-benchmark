"""Miner 1 research: continuous close-location pressure (10d vs 60d), full-library audit."""
# Reuse contemporaneous self-contained all-library audit, changing endpoint and labeling candidate.
src=open('scripts/miner_3_20340511_continuous_close_location_pressure_library_audit.py').read()
src=src.replace("END=pd.Timestamp('2034-05-10')", "END=pd.Timestamp('2034-06-07')")
src=src.replace('continuous close-location pressure.', 'continuous close-location pressure (miner_1 validation).')
src=src.replace("print('AUDIT endpoint'", "print('CANDIDATE continuous_close_location_pressure_10_60 endpoint'")
exec(src)
