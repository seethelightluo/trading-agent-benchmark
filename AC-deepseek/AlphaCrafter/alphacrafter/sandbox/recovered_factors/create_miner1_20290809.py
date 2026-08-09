"""One conditional idea: residual serial-dependence transition active only in cross-asset residual-volatility compression."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_1_20290726_revalidate_residual_serial_dependence_transition_20_60d.py',encoding='utf8').read()
# Reuse the audited data loader and reconstructed admitted-factor panel; update visible cutoff.
src=src.replace("END=pd.Timestamp('2029-07-25')", "END=pd.Timestamp('2029-08-08')")
src=src.replace("# Equal-weight cross-asset residual autocorrelation: change from 60d baseline.\nlag=e.shift(1)\na20=pd.DataFrame({a:e[a].rolling(20,min_periods=14).corr(lag[a]) for a in A})\na60=pd.DataFrame({a:e[a].rolling(60,min_periods=42).corr(lag[a]) for a in A})\nf=a20-a60\nfid='miner_1_residual_serial_dependence_transition_20_60d'", """# Candidate: a recent-vs-structural residual persistence shift, observed only when
# current cross-asset idiosyncratic dispersion is compressed versus its 60d state.
# This avoids applying persistence extrapolation during broad, unstable dispersion shocks.
lag=e.shift(1)
a20=pd.DataFrame({a:e[a].rolling(20,min_periods=14).corr(lag[a]) for a in A})
a60=pd.DataFrame({a:e[a].rolling(60,min_periods=42).corr(lag[a]) for a in A})
transition=a20-a60
csdisp=e.std(axis=1,ddof=0)
compression=csdisp.rolling(20,min_periods=14).mean()/csdisp.rolling(60,min_periods=42).mean()
f=transition.where(compression<1.0, np.nan)
fid='miner_1_compression_conditioned_residual_serial_transition_20_60d'""")
# Regime labels include all historical blocks rather than old narrow labels.
src=src.replace("[('2025_26',ics[5].index<'2027'),('2027_28',(ics[5].index>='2027')&(ics[5].index<'2029')),('2029_onward',ics[5].index>='2029')]", "[('2020_24',ics[5].index<'2025'),('2025_26',(ics[5].index>='2025')&(ics[5].index<'2027')),('2027_28',(ics[5].index>='2027')&(ics[5].index<'2029')),('2029_onward',ics[5].index>='2029')]")
open('scripts/miner_1_20290809_compression_conditioned_residual_serial_transition_20_60d.py','w',encoding='utf8').write(src)
print('written')
