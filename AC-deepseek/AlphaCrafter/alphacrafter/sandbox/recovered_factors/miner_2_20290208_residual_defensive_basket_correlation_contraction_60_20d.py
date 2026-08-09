"""Validate one candidate: residual defensive-basket correlation contraction, 60d vs 20d.
Uses only closes/volumes through the completed 2029-02-07 session."""
exec(open('scripts/miner_3_20290111_residual_defensive_basket_correlation_contraction_60_20d.py').read().replace("END=pd.Timestamp('2029-01-10')", "END=pd.Timestamp('2029-02-07')").replace("miner_3_residual_defensive_basket_correlation_contraction_60_20d", "miner_2_residual_defensive_basket_correlation_contraction_60_20d"))
