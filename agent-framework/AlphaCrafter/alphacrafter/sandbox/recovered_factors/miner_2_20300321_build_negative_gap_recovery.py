# Miner_2 research candidate, current-date refresh.
# Derived from the pre-existing full-library validation template; only cutoff is refreshed.
p='scripts/miner_3_20300307_negative_overnight_gap_intraday_recovery_residual_20.py'
s=open(p,encoding='utf-8').read()
s=s.replace("E=pd.Timestamp('2030-03-06')", "E=pd.Timestamp('2030-03-20')")
s=s.replace('Miner_2: volume-confirmed recovery-quality residual (20 observations), visible through 2029-11-14.', 'Miner_2: negative-overnight-gap intraday recovery residual (20 observations), visible through 2030-03-20.')
s=s.replace("FACTOR negative_overnight_gap_intraday_recovery_residual_20", "FACTOR negative_overnight_gap_intraday_recovery_residual_20")
open('scripts/miner_2_20300321_negative_overnight_gap_intraday_recovery_residual_20.py','w',encoding='utf-8').write(s)
print('written')
