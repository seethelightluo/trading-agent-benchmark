"""miner_2 single-idea validation: downside opening-gap fill resilience residual (20 sessions)."""
from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-06-25')")
old="""# Candidate: difference in 30-day WTI beta observed on broad-market down versus non-down sessions.
# Higher scores indicate relative oil-shock transmission strength during market stress,
# residualized from generic risk, crowding, beta asymmetry and trend.
fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""# Candidate: downside opening-gap fill resilience. For each instrument, identify
# sessions opening below the prior close, scale its close-to-open return by the
# absolute opening gap, and average over the past 20 sessions. Higher values
# indicate more complete recovery of adverse overnight gaps. Residualize against
# realized volatility, peer crowding, downside market-beta asymmetry and trend.
O=pd.DataFrame({a:rd(a,'open') for a in A})
gap=O/P.shift(1)-1
intr=P/O-1
raw=pd.DataFrame({a:(intr[a]/(-gap[a]).clip(lower=0.002)).where(gap[a]<-0.002).rolling(20,min_periods=5).mean() for a in A})
F=res(raw,v,peer,dba,trend)"""
if old not in src: raise RuntimeError('candidate anchor absent')
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','downside_opening_gap_fill_resilience_residual_20')
src=src.replace("for n,m in []:\n x=ics[1][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')", "for n,m in [('2020_23',ics[5].index<'2024-01-01'),('2024_27',(ics[5].index>='2024-01-01')&(ics[5].index<'2028-01-01')),('2028_current',ics[5].index>='2028-01-01')]:\n x=ics[5][m];print('regime5',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')")
exec(compile(src,'miner_2_gap_fill_resilience_20310626','exec'))
