"""One candidate: inverse residual sustained VIX-relief fragility, point-in-time validation."""
src=open('scripts/miner_3_20290208_residual_sustained_vix_relief_20_80obs.py').read()
src=src.replace('"""One candidate: residual multi-horizon VIX-relief resilience, validated point-in-time."""','"""One candidate: inverse residual sustained VIX-relief fragility, point-in-time validation."""')
src=src.replace("END=pd.Timestamp('2029-02-07')", "END=pd.Timestamp('2029-02-21')")
src=src.replace("# Candidate rewards sustained relief performance relative to immediate relief performance; residualization makes it incremental to the admitted 40d relief signal.\nshort=", "# Candidate is the inverse sustained-minus-immediate relief spread: assets whose recent VIX-relief return is unusually strong versus 80d relief history are fragile mean-reversion candidates.\nshort=")
src=src.replace("f=resid(raw,[vd,vu,es,down,kurt,trend,du,dd])", "f=-resid(raw,[vd,vu,es,down,kurt,trend,du,dd])")
open('scripts/miner_3_20290222_inverse_residual_sustained_vix_relief_fragility_20_80obs.py','w').write(src)
print('wrote candidate')
