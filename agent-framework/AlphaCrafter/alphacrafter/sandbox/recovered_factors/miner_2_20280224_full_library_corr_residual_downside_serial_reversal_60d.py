"""Exhaustive-current-library correlation recheck for residual downside serial reversal."""
p='scripts/miner_2_20271216_revalidate_drawdown_synchronization_60_20.py'
s=open(p).read().replace("END=pd.Timestamp('2027-12-15')", "END=pd.Timestamp('2028-02-23')")
s=s.replace("sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});f=sy.shift(20)-sy", "sy=pd.DataFrame({a:r[a].rolling(60,min_periods=40).corr(breadth.diff()) for a in A});down=e.clip(upper=0)\ndef ac(x):\n z=pd.DataFrame({'x':x,'lag':x.shift(1)}).dropna()\n return z.x.corr(z.lag) if len(z)>=45 else np.nan\nf=-down.rolling(60,min_periods=45).apply(ac,raw=False)")
s=s.replace("FACTOR drawdown_sync_improvement_60_20", "FACTOR residual_downside_serial_reversal_60d")
s=s.replace("lib['miner_2_drawdown_synchronization_improvement_60_20']=f", "lib['miner_2_drawdown_synchronization_improvement_60_20']=sy.shift(20)-sy")
s=s.replace("lib['miner_2_residual_downside_serial_reversal_60d']=loss.rolling(20,min_periods=12).mean()-loss.shift(20).rolling(20,min_periods=12).mean()", "lib['miner_2_residual_downside_serial_reversal_60d']=f")
s=s.replace("if name!= 'miner_2_drawdown_synchronization_improvement_60_20'", "if name!= 'miner_2_residual_downside_serial_reversal_60d'")
s=s.replace("if n!='miner_2_drawdown_synchronization_improvement_60_20'", "if n!='miner_2_residual_downside_serial_reversal_60d'")
exec(s)
