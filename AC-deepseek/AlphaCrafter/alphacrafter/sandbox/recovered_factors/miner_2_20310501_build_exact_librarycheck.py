from pathlib import Path
src=Path('scripts/miner_3_20301003_oil_market_drawdown_conditional_transmission_residual_30.py').read_text()
src=src.replace("E=pd.Timestamp('2030-10-02')", "E=pd.Timestamp('2031-04-30')")
old="""fx=R['WTI']
F=res(beta(fx,M<0)-beta(fx,M>=0),v,peer,dba,trend)"""
new="""x=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(lower=0)
stress=x/(1+x); rate=R['US10Y']
F=-res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)
stress_old=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3
old_factor=-res(beta(rate*stress_old,pd.Series(True,index=P.index))-beta(rate*(1-stress_old),pd.Series(True,index=P.index)),v,peer,dba,trend)"""
src=src.replace(old,new).replace('oil_market_drawdown_conditional_transmission_residual_30','smooth_equity_stress_rate_transmission_residual_30')
src=src.replace("for nm,z in [('dxy','DXY',1)","L['inverse_equity_stress_amplified_rate_transmission_residual_30']=old_factor\nfor nm,z in [('dxy','DXY',1)")
Path('scripts/miner_2_20310501_exact_librarycheck.py').write_text(src)
