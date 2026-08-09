"""Revalidate one admitted factor: residual VIX-DXY stress-transition, PIT through 2029-05-30."""
exec(open('scripts/miner_3_20290503_residual_vix_dxy_stress_transition_60obs.py').read().replace("END=pd.Timestamp('2029-05-02')", "END=pd.Timestamp('2029-05-30')"))
