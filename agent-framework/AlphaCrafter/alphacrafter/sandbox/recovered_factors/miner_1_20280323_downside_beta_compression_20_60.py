"""Validate one candidate: downside beta compression (20 vs 60 observations).
An asset whose market-downside beta has fallen relative to its longer baseline may
be becoming a more useful defensive cross-asset holding.  The embedded harness
also tests novelty against all currently admitted 19 library signals.
"""
from pathlib import Path
source = Path('scripts/miner_1_20280309_downside_peer_correlation_dispersion_resilience_40.py').read_text()
source = source.replace("cut=pd.Timestamp('2028-03-08')", "cut=pd.Timestamp('2028-03-22')")
old = "P=pd.DataFrame(C);r=P.pct_change();neg=r.where(r<0,0.)\nshort=np.sqrt((neg**2).rolling(10,min_periods=7).mean());long=np.sqrt((neg**2).rolling(40,min_periods=25).mean()); raw=-np.log((short+1e-5)/(long+1e-5));f=raw.sub(raw.median(axis=1),axis=0);fw={h:P.shift(-h)/P-1 for h in H}"
new = "P=pd.DataFrame(C);r=P.pct_change();m=r.median(axis=1); down=r.where(m<0); var20=m.where(m<0).rolling(20,min_periods=10).var();var60=m.where(m<0).rolling(60,min_periods=25).var(); b20=pd.DataFrame({a:down[a].rolling(20,min_periods=10).cov(m.where(m<0))/var20 for a in A});b60=pd.DataFrame({a:down[a].rolling(60,min_periods=25).cov(m.where(m<0))/var60 for a in A});raw=-(b20-b60);f=raw.sub(raw.median(axis=1),axis=0);fw={h:P.shift(-h)/P-1 for h in H}"
assert old in source
source = source.replace(old, new)
source = source.replace("FACTOR peer_relative_downside_volatility_compression_10_40", "FACTOR downside_beta_compression_20_60")
Path('scripts/miner_1_20280323_downside_beta_compression_20_60.py').write_text(source)
exec(compile(source, 'downside_beta_compression_20_60_harness.py', 'exec'))
