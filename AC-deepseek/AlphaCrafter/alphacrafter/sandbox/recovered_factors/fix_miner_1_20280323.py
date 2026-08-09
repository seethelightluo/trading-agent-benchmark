from pathlib import Path
p=Path('scripts/miner_1_20280323_downside_beta_compression_20_60.py')
s=p.read_text()
s=s.replace("md=m.where(m<0);rd=r.where(m<0,axis=0)","md=m.where(m<0);rd=r.where(pd.DataFrame({a:m<0 for a in A}))")
s=s.replace("rolling(20,min_periods=10)","rolling(40,min_periods=10)").replace("rolling(60,min_periods=25)","rolling(120,min_periods=25)")
s=s.replace("downside_beta_compression_20_60", "downside_beta_compression_40_120")
p.write_text(s)
