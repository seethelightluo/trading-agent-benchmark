"""Revalidation wrapper for drawdown-sync factor through 2028-02-23.
Uses the established full-history construction and reports current admission metrics."""
exec(open('scripts/miner_2_20271216_revalidate_drawdown_synchronization_60_20.py').read().replace("END=pd.Timestamp('2027-12-15')", "END=pd.Timestamp('2028-02-23')").replace("updated through 2027-11-03", "updated through 2028-02-23"))
