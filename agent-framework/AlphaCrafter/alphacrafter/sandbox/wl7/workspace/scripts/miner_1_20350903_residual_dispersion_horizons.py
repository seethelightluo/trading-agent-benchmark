# Reuse the residual-dispersion reversal construction and compare forward horizons.
exec(open('scripts/miner_1_20350903_residual_dispersion_reversal.py').read().split("fr=np.log(px.shift(-20)/px)")[0])
for h in (5,10,20):
 fr=np.log(px.shift(-h)/px).replace([np.inf,-np.inf],np.nan); vals=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   c=a[ok].corr(b[ok])
   if np.isfinite(c): vals.append(c)
 q=np.array(vals); print('horizon',h,'observations',len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),np.mean(q>0)))
