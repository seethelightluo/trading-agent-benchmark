p='scripts/miner_1_20290906_vix_residual.py'
s=open(p).read()
s=s.replace("if len(q)>=8:\n  rows.append((P.index[t],q.iloc[:,0].corr(q.iloc[:,1]),len(q)));sigs.append(pd.Series(vals,name=P.index[t]))", "if len(q)>=8 and q.iloc[:,0].std()>1e-12 and q.iloc[:,1].std()>1e-12:\n  ic=q.iloc[:,0].corr(q.iloc[:,1])\n  if np.isfinite(ic): rows.append((P.index[t],ic,len(q)));sigs.append(pd.Series(vals,name=P.index[t]))")
s=s.replace("if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))", "if len(q)>=8 and q.iloc[:,0].std()>1e-12 and q.iloc[:,1].std()>1e-12:\n   ic=q.iloc[:,0].corr(q.iloc[:,1])\n   if np.isfinite(ic): z.append(ic)")
open(p,'w').write(s)
