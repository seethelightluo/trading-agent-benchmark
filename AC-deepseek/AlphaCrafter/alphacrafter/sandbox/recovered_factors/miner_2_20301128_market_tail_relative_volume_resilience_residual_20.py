"""Miner_2 single-idea validation: market-tail relative-volume resilience residual."""
from pathlib import Path
src=Path('scripts/miner_2_20301003_persistent_market_stress_correlation_asymmetry_residual_60.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2030-11-27')")
src=src.replace('persistent_market_stress_correlation_asymmetry_residual_60','market_tail_relative_volume_resilience_residual_20')
old="""# Only use sessions after a persisting three-session market drawdown, then compare co-movement to ordinary conditions.
m3=(1+M).rolling(3,min_periods=3).apply(np.prod,raw=True)-1
stressmask=m3.le(m3.rolling(60,min_periods=40).quantile(.20))
stresscorr=pd.DataFrame({a:R[a].where(stressmask).rolling(60,min_periods=8).corr(M.where(stressmask)) for a in A})
allcorr=pd.DataFrame({a:R[a].rolling(60,min_periods=40).corr(M) for a in A})
F=res(-(stresscorr-allcorr),v,peer,dba,trend)"""
new="""# Candidate: resilience on high-participation market-tail days.  Assets with less
# negative (or positive) return while their own volume is unusually high during a
# cross-asset tail are preferred; use only lagged/finished sessions and residualize
# standard risk, crowding, downside beta, and trend exposures.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
rv=V/V.rolling(20,min_periods=15).mean()-1
stressmask=M.le(M.rolling(60,min_periods=40).quantile(.20))
w=rv.clip(lower=0).where(stressmask).fillna(0)
raw=R.mul(w).rolling(20,min_periods=15).sum().div(w.rolling(20,min_periods=15).sum().replace(0,np.nan),axis=0)/(v+1e-12)
F=res(raw,v,peer,dba,trend)
allcorr=pd.DataFrame({a:R[a].rolling(60,min_periods=40).corr(M) for a in A})"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new)
exec(compile(src,'market_tail_relative_volume_resilience_20301128','exec'))
