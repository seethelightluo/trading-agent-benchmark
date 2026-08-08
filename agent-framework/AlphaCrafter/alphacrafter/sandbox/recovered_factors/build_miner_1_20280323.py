from pathlib import Path
source=Path('scripts/miner_1_20280309_downside_peer_correlation_dispersion_resilience_40.py').read_text()
source=source.replace("cut=pd.Timestamp('2028-03-08')", "cut=pd.Timestamp('2028-03-22')")
a=source.index("P=pd.DataFrame(C);r=P.pct_change();m=r.median(axis=1); down=m<0")
b=source.index("def ev(h,span=None):")
replacement="""P=pd.DataFrame(C);r=P.pct_change();m=r.median(axis=1)
# Conditional beta in broad-down days; negative short-minus-long favors beta compression.
md=m.where(m<0);rd=r.where(m<0,axis=0)
v20=md.rolling(20,min_periods=10).var();v60=md.rolling(60,min_periods=25).var()
b20=pd.DataFrame({a:rd[a].rolling(20,min_periods=10).cov(md)/v20 for a in A})
b60=pd.DataFrame({a:rd[a].rolling(60,min_periods=25).cov(md)/v60 for a in A})
f=(-(b20-b60)).sub((-(b20-b60)).median(axis=1),axis=0);fw={h:P.shift(-h)/P-1 for h in H}
"""
source=source[:a]+replacement+source[b:]
source=source.replace('downside_peer_correlation_dispersion_resilience_40 cutoff','downside_beta_compression_20_60 cutoff')
source=source.replace("('2027_28',('2027-01-01','2028-03-08'))", "('2027_28',('2027-01-01','2028-03-22'))")
Path('scripts/miner_1_20280323_downside_beta_compression_20_60.py').write_text(source)
