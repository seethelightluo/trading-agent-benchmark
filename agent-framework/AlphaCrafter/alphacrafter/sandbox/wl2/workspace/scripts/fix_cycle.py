p='scripts/miner_3_20321028_vix_breadth_capitulation.py';s=open(p).read();s=s.replace("C=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index(); R=C.pct_change()","C=pd.DataFrame({s:z.set_index(pd.to_datetime(z.date)).close.astype(float) for s,z in D.items()}).sort_index(); C=C.groupby(level=0).last(); R=C.pct_change()")
# remove noisy diagnostic
s=s.replace(";\n if len(q)<8 and active.loc[d]>0: print('diag',d,len(q),int(f.loc[d].notna().sum()),int(R.shift(-1).loc[d].notna().sum()))","")
open(p,'w').write(s)
