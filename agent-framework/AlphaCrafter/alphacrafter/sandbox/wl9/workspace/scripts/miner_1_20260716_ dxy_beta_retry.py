# aligned DXY beta retry
exec(open('scripts/miner_1_20260716_dxy_beta.py').read().split('for window')[0])
for w in [40,60,120]:
 fac=pd.DataFrame(index=R.index)
 mm=m.rolling(w).mean(); vv=((m-mm)**2).rolling(w).mean()
 for s in prices:
  rm=R[s].rolling(w).mean(); cv=((R[s]-rm)*(m-mm)).rolling(w).mean(); fac[s]=-cv/vv
 fwd=R.shift(-1); a=[]; ns=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a); print(w,len(a),np.mean(ns) if ns else 0,np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0) if len(a) else 0,fac.notna().mean().mean())
