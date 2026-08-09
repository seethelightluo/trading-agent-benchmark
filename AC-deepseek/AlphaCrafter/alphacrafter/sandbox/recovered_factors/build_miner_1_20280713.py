"""One idea: dispersion-shock gated peer-relative reversal."""
p='scripts/miner_3_20280629_conditional_downside_participation_avoidance_60.py'
s=open(p).read()
a=s.index('# Defensive resilience:')
b=s.index('fw={h:',a)
new='''# A high, rising cross-asset return-dispersion regime signals heterogeneous
# dislocations. Prefer recent peer-relative losers, volatility-normalized; retain
# event evidence for 20 observations to avoid a one-day timing dependency.
disp=r.std(axis=1)
shock=(disp>disp.rolling(60,min_periods=40).quantile(.75)) & (disp>disp.shift(5))
peer=r.sub(r.median(axis=1),axis=0)
raw=-peer.rolling(5,min_periods=4).sum()/r.rolling(20,min_periods=15).std()
f=raw.where(shock,axis=0).rolling(20,min_periods=5).mean()
f=f.sub(f.median(axis=1),axis=0)
'''
s=s[:a]+new+s[b:]
s=s.replace('conditional_downside_participation_avoidance_60','dispersion_shock_peer_reversal_20')
open('scripts/miner_1_20280713_dispersion_shock_peer_reversal_20.py','w').write(s)
