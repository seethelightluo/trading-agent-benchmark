"""miner_2 one-idea test: signed volume participation asymmetry residual (20 sessions)."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-06-25')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: signed volume-participation asymmetry. Volume is normalized by its
# own 20-session median; compute the 20-session mean normalized participation on
# positive-return sessions minus that on negative-return sessions. A high score
# identifies instruments for which demand participation dominates supply
# participation, independent of realized volatility, peer crowding, downside
# market-beta asymmetry and 20-session trend.
V=pd.DataFrame({a:rd(a,'volume') for a in A})
rv=V/V.rolling(20,min_periods=12).median().replace(0,np.nan)
raw=pd.DataFrame({a:rv[a].where(R[a]>0).rolling(20,min_periods=6).mean()-rv[a].where(R[a]<0).rolling(20,min_periods=6).mean() for a in A})
F=res(raw,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','signed_volume_participation_asymmetry_residual_20')
# add additional regime split, preserving original output.
src=src.replace("for n,m in []:\n x=ics[1][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')", "for n,m in [('2020_23',ics[5].index<'2024-01-01'),('2024_27',(ics[5].index>='2024-01-01')&(ics[5].index<'2028-01-01')),('2028_current',ics[5].index>='2028-01-01')]:\n x=ics[5][m];print('regime5',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')")
exec(compile(src,'miner_2_signed_volume_participation_20310626','exec'))
