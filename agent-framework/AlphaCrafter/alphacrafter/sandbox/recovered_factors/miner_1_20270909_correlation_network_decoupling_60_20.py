"""Validate asset-specific correlation-network decoupling: one interpretable idea."""
import pathlib
src=pathlib.Path('scripts/miner_1_20270826_dispersion_shock_residual_resilience_60d.py').read_text()
src=src.replace("END=pd.Timestamp('2027-08-25')", "END=pd.Timestamp('2027-09-08')")
old="""# On days with unusually wide cross-asset return dispersion, measure each asset's
# market-residual return.  A high score indicates repeated idiosyncratic resilience
# rather than merely a high beta response during turbulent cross-asset selection.
disp=r.std(axis=1); shock=disp>=disp.rolling(60,min_periods=40).quantile(.75)
f=e.where(shock,np.nan).rolling(60,min_periods=12).mean()/e.rolling(60,min_periods=40).std()"""
new="""# Asset-specific correlation-network decoupling.  For each asset, compare its
# mean pairwise correlation to the other 14 assets over 20 versus 60 sessions.
# A high score means recently becoming less network-connected (more idiosyncratic).
peer20=pd.DataFrame(index=p.index,columns=A,dtype=float)
peer60=pd.DataFrame(index=p.index,columns=A,dtype=float)
for a in A:
    others=[b for b in A if b!=a]
    peer20[a]=pd.concat([r[a].rolling(20,min_periods=15).corr(r[b]) for b in others],axis=1).mean(axis=1)
    peer60[a]=pd.concat([r[a].rolling(60,min_periods=40).corr(r[b]) for b in others],axis=1).mean(axis=1)
f=peer60-peer20"""
src=src.replace(old,new)
src=src.replace("'shock_days',int(shock.sum())", "'network_windows','20_vs_60'")
src=src.replace("FACTOR dispersion_shock_residual_resilience_60d", "FACTOR correlation_network_decoupling_60_20")
# Deprecated downside-beta is not an admitted library member.
src=src.replace("lib['miner_2_downside_beta_improvement_120_20']=pd.DataFrame(db120-db20,index=p.index,columns=A)\n", "")
pathlib.Path('scripts/miner_1_20270909_correlation_network_decoupling_60_20.py').write_text(src)
print('written')
