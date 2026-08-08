"""Miner_2 one candidate: dispersion-conditioned close-location resilience residual."""
from pathlib import Path
src=Path('scripts/miner_2_20301003_persistent_market_stress_correlation_asymmetry_residual_60.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-02-19')")
src=src.replace('persistent_market_stress_correlation_asymmetry_residual_60','dispersion_conditioned_close_location_resilience_residual_20')
old="""# Only use sessions after a persisting three-session market drawdown, then compare co-movement to ordinary conditions.
m3=(1+M).rolling(3,min_periods=3).apply(np.prod,raw=True)-1
stressmask=m3.le(m3.rolling(60,min_periods=40).quantile(.20))
stresscorr=pd.DataFrame({a:R[a].where(stressmask).rolling(60,min_periods=8).corr(M.where(stressmask)) for a in A})
allcorr=pd.DataFrame({a:R[a].rolling(60,min_periods=40).corr(M) for a in A})
F=res(-(stresscorr-allcorr),v,peer,dba,trend)"""
new="""# Candidate: close-location resilience emphasized only when cross-asset dispersion is elevated.
# Location is (close-low)/(high-low); 20d loss-day mean minus gain-day mean measures
# ability to finish weak sessions away from their lows. The continuous multiplier is
# the non-negative 60d z-score of completed cross-sectional return dispersion.
O=pd.DataFrame({a:rd(a,'open') for a in A});H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A})
loc=(P-Lo)/(H-Lo).replace(0,np.nan)
base=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
disp=R.std(axis=1); dz=((disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)).clip(lower=0,upper=3)
raw=base.mul(1+dz,axis=0)
F=res(raw,v,peer,dba,trend)
allcorr=pd.DataFrame({a:R[a].rolling(60,min_periods=40).corr(M) for a in A})"""
if old not in src: raise RuntimeError('anchor')
src=src.replace(old,new)
src=src.replace("for n,m in [('2020_21',ics[1].index<'2022-01-01'),('2022_23',(ics[1].index>='2022-01-01')&(ics[1].index<'2024-01-01')),('2024_25',(ics[1].index>='2024-01-01')&(ics[1].index<'2026-01-01')),('2026_27',(ics[1].index>='2026-01-01')&(ics[1].index<'2028-01-01')),('2028_ytd',ics[1].index>='2028-01-01')]:\n x=ics[1][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')", "for n,m in [('2026_27',(ics[5].index>='2026-01-01')&(ics[5].index<'2028-01-01')),('2028_ytd',ics[5].index>='2028-01-01')]:\n x=ics[5][m];print('h5_regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')")
exec(compile(src,'miner_2_dispersion_location_20310220','exec'))
