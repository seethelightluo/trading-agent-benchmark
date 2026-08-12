p='scripts/miner_3_20321028_vix_breadth_capitulation.py';s=open(p).read();s=s.replace("for d in f.index:\n q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()", "for i,d in enumerate(f.index[:-1]):\n q=pd.concat([f.iloc[i],R.iloc[i+1]],axis=1).dropna()")
s=s.replace("for d in f.index:\n  q=pd.concat([f.loc[d],rr.loc[d]],axis=1).dropna()", "for i,d in enumerate(f.index[:-h]):\n  q=pd.concat([f.iloc[i],rr.iloc[i+h]],axis=1).dropna()")
open(p,'w').write(s)
