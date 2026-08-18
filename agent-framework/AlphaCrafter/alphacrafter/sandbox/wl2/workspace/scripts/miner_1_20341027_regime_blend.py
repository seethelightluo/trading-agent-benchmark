import pandas as pd,numpy as np
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-10-26'); C={}
for s in S:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index();C[s]=x.loc[x.index<=E,'close']
C=pd.DataFrame(C); R=C.pct_change(); br=(R.rolling(20).sum()>0).mean(axis=1)
mom=C.pct_change(60); rv=-C.pct_change(5); v=R.rolling(20,min_periods=15).std()*np.sqrt(252)
w=(br-0.5)*2; w=w.clip(-1,1)
F=mom.mul(w,axis=0)+rv.mul((1-w.abs()),axis=0); F=F.div(v).shift(1)
F.loc[F.notna().sum(axis=1)>=8].reset_index().to_csv('../persistent/miner_1_20341027_regime_blend_signal.csv',index=False)
for h in [1,5,10,20,40]:
 fr=C.shift(-h)/C-1; a=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(a);print(h,len(a),round(np.mean(ns),2),round(np.nanmean(a),6),round(np.nanmean(a)/np.nanstd(a,ddof=1),6),round(np.mean(a>0),4))
print('cov',round(F.notna().sum().sum()/(len(F)*15),5),'turn',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).median(),5))
