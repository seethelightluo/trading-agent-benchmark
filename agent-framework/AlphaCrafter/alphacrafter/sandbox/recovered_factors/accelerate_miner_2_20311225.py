from pathlib import Path
p=Path('scripts/miner_2_20311225_gap_fade_efficiency_fullcorrected.py')
s=p.read_text()
old="peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})"
new="""# Algebraic rolling correlations are materially faster than repeated pandas rolling.corr calls.
def rcorr(x,y,w=20,mp=15):
    mx=x.rolling(w,min_periods=mp).mean(); my=y.rolling(w,min_periods=mp).mean()
    cv=(x*y).rolling(w,min_periods=mp).mean()-mx*my
    vx=(x*x).rolling(w,min_periods=mp).mean()-mx*mx; vy=(y*y).rolling(w,min_periods=mp).mean()-my*my
    return cv/(np.sqrt(vx*vy)+1e-12)
peer=pd.DataFrame({a:pd.concat([rcorr(R[a],R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})"""
assert old in s
p2=Path('scripts/miner_2_20311225_gap_fade_efficiency_fast.py')
p2.write_text(s.replace(old,new))
print(p2)
