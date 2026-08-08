p='scripts/miner_1_20320415_residual_downside_gap_repair_acceleration_20_60obs.py'
s=open(p).read().replace("event=resid.shift(1)<-resid.rolling(60,min_periods=45).std().shift(1)","event=resid.shift(1)<0")
s=s.replace('conditional on previous residual downside shock','conditional on any previous residual downside day')
open(p,'w').write(s)
