"""Revalidate one factor only: inverse residual-downside normalized range exhaustion."""
p='scripts/miner_1_20310403_inverse_residual_downside_range_expansion_exhaustion_60obs.py'
s=open(p).read().replace("END=pd.Timestamp('2031-04-02')", "END=pd.Timestamp('2031-05-28')")
# Revalidation does not need to reconstruct historical library panels; retain all core factor calculations/diagnostics.
s=s[:s.index('# Complete 30-signal reconstruction')]
open('scripts/miner_1_20310529_revalidate_inverse_residual_downside_range_expansion_exhaustion_60obs.py','w').write(s)
print('wrote revalidation script')
