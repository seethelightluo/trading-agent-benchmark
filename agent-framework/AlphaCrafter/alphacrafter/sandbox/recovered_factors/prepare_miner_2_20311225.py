from pathlib import Path
p=Path('scripts/miner_2_20311127_gap_fade_efficiency_residual_20.py');s=p.read_text()
s=s.replace("E=pd.Timestamp('2031-11-26')", "E=pd.Timestamp('2031-12-24')")
# Correct forward returns so they use each instrument's next h observed sessions, not the union calendar.
s=s.replace("out=[];nn=[];fw=P.shift(-h)/P-1", "out=[];nn=[];fw=pd.DataFrame({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index) for a in A})")
s=s.replace("print('FACTOR gap_fade_efficiency_residual_20 visible_through',E.date(),'assets',len(A),'library_signals',len(L))", "print('FACTOR gap_fade_efficiency_residual_20 visible_through',E.date(),'assets',len(A),'library_signals',len(L),flush=True)")
Path('scripts/miner_2_20311225_gap_fade_efficiency_fullcorrected.py').write_text(s)
print('written')
