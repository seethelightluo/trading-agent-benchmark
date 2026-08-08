p='scripts/miner_3_20290322_vix_shock_transmission_beta_asymmetry_residual_30.py'
s=open(p).read()
needle="src=src.replace(old,new)\nexec(compile(src,'vix_shock_transmission_candidate','exec'))"
repl="""src=src.replace(old,new)
# Include the current Miner_2 admitted yield-hedge signal in the mandatory library correlation screen.
needle2=\"L['market_down_intraday_recovery_residual_20']=res(intr,mom,v,peer)\"
addition=\"\"\"L['market_down_intraday_recovery_residual_20']=res(intr,mom,v,peer)
# admitted 2029-03-08: downside-market US10Y transmission residual
_y=R['US10Y']; _md=M<0
_bd=pd.DataFrame({a:R[a].where(_md).rolling(30,min_periods=10).cov(_y.where(_md))/_y.where(_md).rolling(30,min_periods=10).var() for a in A})
_bn=pd.DataFrame({a:R[a].where(~_md).rolling(30,min_periods=10).cov(_y.where(~_md))/_y.where(~_md).rolling(30,min_periods=10).var() for a in A})
L['downside_market_yield_hedge_beta_residual_30']=res(_bd-_bn,v,peer,dba,P/P.shift(20)-1)\"\"\"
if needle2 not in src: raise RuntimeError('library anchor missing')
src=src.replace(needle2,addition)
exec(compile(src,'vix_shock_transmission_candidate','exec'))"""
if needle not in s: raise RuntimeError('wrapper anchor missing')
open(p,'w').write(s.replace(needle,repl))
