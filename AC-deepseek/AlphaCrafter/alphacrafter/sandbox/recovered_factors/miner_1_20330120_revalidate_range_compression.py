"""Scheduled revalidation of the admitted residual downside range-compression signal.
Uses only completed bars through 2033-01-19; forward returns are research labels."""
exec(open('scripts/miner_1_20320304_revalidate_residual_downside_range_compression.py').read().replace("END=pd.Timestamp('2032-03-03')", "END=pd.Timestamp('2033-01-19')"))
