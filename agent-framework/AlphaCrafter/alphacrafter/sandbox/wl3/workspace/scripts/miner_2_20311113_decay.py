# append decay diagnostics
for h in [5,10,20]:
 yy=np.log(p.shift(-h)/p); vals=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: vals.append(a[ok].corr(b[ok]))
 print('H%d IC %.8f ICIR %.8f'%(h,np.nanmean(vals),np.nanmean(vals)/np.nanstd(vals,ddof=1)))
