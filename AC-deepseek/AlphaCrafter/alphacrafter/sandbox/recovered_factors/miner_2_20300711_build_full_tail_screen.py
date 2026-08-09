from pathlib import Path
p=Path('scripts/miner_2_20300627_tail_correlation_asymmetry_residual_60.py')
s=p.read_text()
s=s.replace("E=pd.Timestamp('2030-06-26')", "E=pd.Timestamp('2030-07-10')")
s=s.replace('visible_through', 'visible_through')
add="""
# Remaining admitted signals: reconstructed from their persisted definitions for complete 29-file screen.
_comm=R[['XAU','COPPER','WTI']].mean(axis=1)-R[['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX']].mean(axis=1)
_cb=pd.DataFrame({a:R[a].rolling(30,min_periods=12).cov(_comm)/_comm.rolling(30,min_periods=12).var() for a in A})
L['commodity_equity_divergence_beta_residual_30']=res(_cb,beta(pd.Series(True,index=P.index)),v,peer)
_V=pd.DataFrame({a:rd(a,'volume') for a in A}); _vratio=_V/_V.rolling(20,min_periods=15).mean(); _q75=_vratio.rolling(60,min_periods=40).quantile(.75)
_liqraw=pd.DataFrame({a:R[a].where(_vratio[a]>=_q75[a]).rolling(20,min_periods=6).mean()/(v[a]+1e-12) for a in A})
L['inverse_liquidity_shock_resilience_residual_20']=-res(_liqraw,P/P.shift(20)-1,v,peer)
_sev=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,4); _rel=_vratio.clip(.5,2).where(_vratio.notna(),1.0); _pw=_sev*_rel
_praw=R.mul(_pw).rolling(20,min_periods=10).sum().div(_pw.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12)
L['continuous_participation_weighted_rebound_residual_20']=res(_praw,v,peer,dba,P/P.shift(20)-1)
_mdloc=loc.where((R<0).mul(M<0,axis=0),np.nan).rolling(20,min_periods=6).mean()
L['market_down_loss_close_location_residual_20']=res((_mdloc-.5)/(v+1e-12),v,peer,P/P.shift(20)-1,cl)
_rng=(H-Lo).div(P.shift(1)).replace([np.inf,-np.inf],np.nan); _rngrel=_rng.div(_rng.rolling(20,min_periods=15).mean()+1e-12)
_rngraw=((loc-.5)*_rngrel).where((R<0).mul(M<0,axis=0),np.nan).rolling(20,min_periods=6).mean()/(v+1e-12)
L['market_down_range_expansion_recovery_residual_20']=res(_rngraw,v,peer,P/P.shift(20)-1,cl,_mdloc)
"""
s=s.replace("print('FACTOR tail_correlation_asymmetry_residual_60",add+"\nprint('FACTOR tail_correlation_asymmetry_residual_60")
Path('scripts/miner_2_20300711_tail_correlation_asymmetry_residual_60_full_library.py').write_text(s)
print('written')
