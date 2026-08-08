"""Revalidate one admitted factor PIT through 2029-04-18."""
exec(open('scripts/miner_3_20290405_revalidate_inverse_residual_vix_relief_fragility_20_80.py').read().replace("END=pd.Timestamp('2029-04-04')", "END=pd.Timestamp('2029-04-18')").replace("PIT through 2029-04-04", "PIT through 2029-04-18"))
