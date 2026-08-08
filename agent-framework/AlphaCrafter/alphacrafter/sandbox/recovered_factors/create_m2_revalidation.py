# Revalidation script created by adapting the prior panel/diagnostic harness.
p='scripts/miner_2_20270826_revalidate_drawdown_synchronization_60_20.py'
s=open(p).read()
s=s.replace("END=pd.Timestamp('2027-08-25')", "END=pd.Timestamp('2027-09-08')")
s=s.replace("# Lag-one autocorrelation of the downside component of market-neutral residual returns.\n# Setting non-downside observations to zero preserves a common 60-day history while\n# isolating the temporal persistence of idiosyncratic losses.\nneg=e.clip(upper=0)\ndd0=p/p.rolling(60,min_periods=40).max()-1; breadth0=(dd0<-.05).mean(axis=1); sy0=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth0.diff()) for a in A}); f=sy0.shift(20)-sy0", "# Candidate: increase in each asset's trailing correlation with the equal-weight cross-asset market return.\n# A high reading identifies rising common-factor synchronization over the last 20 sessions.\nmc0=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(m) for a in A}); f=mc0-mc0.shift(20)")
s=s.replace("print('FACTOR revalidate_drawdown_synchronization_improvement_60_20'", "print('FACTOR revalidate_market_synchronization_increase_60_20'")
s=s.replace("lib['miner_2_market_synchronization_increase_60_20']=mc-mc.shift(20);", "# candidate deliberately excluded from comparison library\n")
open('scripts/miner_2_20270909_revalidate_market_synchronization_60_20.py','w').write(s)
print('written')
