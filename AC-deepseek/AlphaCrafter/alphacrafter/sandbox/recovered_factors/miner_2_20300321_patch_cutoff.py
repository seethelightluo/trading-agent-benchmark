p='scripts/miner_2_20300321_negative_overnight_gap_intraday_recovery_residual_20.py'
s=open(p).read().replace("E=pd.Timestamp('2030-03-20')", "E=pd.Timestamp('2030-02-28')").replace('visible through 2030-03-20','visible through 2030-02-28')
open(p,'w').write(s)
