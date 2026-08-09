p='scripts/miner_2_20300321_negative_overnight_gap_intraday_recovery_residual_20.py';s=open(p).read()
s=s.replace("d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')", "d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[c],errors='coerce')")
s=s.replace('for t in P.index:\n  q=pd.concat([F.loc[t].rename(\'f\'),fw.loc[t].rename(\'r\')]', 'for t in P.index[P.index<=E]:\n  q=pd.concat([F.loc[t].rename(\'f\'),fw.loc[t].rename(\'r\')]')
open(p,'w').write(s)
