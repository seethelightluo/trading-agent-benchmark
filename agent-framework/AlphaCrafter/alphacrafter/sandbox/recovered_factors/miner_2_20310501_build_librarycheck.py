from pathlib import Path
src=Path('scripts/miner_2_20310501_smooth_equity_stress_rate_transmission_residual_30.py').read_text()
needle="F=-res(beta(rate*stress,pd.Series(True,index=P.index))-beta(rate*(1-stress),pd.Series(True,index=P.index)),v,peer,dba,trend)"
repl=needle+"\n# Include the April admitted predecessor in the library screen.\nstress_old=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,3)/3\nold_factor=-res(beta(rate*stress_old,pd.Series(True,index=P.index))-beta(rate*(1-stress_old),pd.Series(True,index=P.index)),v,peer,dba,trend)"
src=src.replace(needle,repl)
src=src.replace("L={'dxy_beta_residual_peer20'", "L={'inverse_equity_stress_amplified_rate_transmission_residual_30':old_factor,'dxy_beta_residual_peer20'")
Path('scripts/miner_2_20310501_librarycheck.py').write_text(src)
