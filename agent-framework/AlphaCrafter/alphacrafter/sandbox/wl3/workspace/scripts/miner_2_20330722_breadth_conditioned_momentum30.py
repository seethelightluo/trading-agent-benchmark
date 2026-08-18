import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
mom=lp.diff(30); rel=mom.subtract(mom.median(axis=1),axis='index'); rv=r.rolling(30,min_periods=20).std()*np.sqrt(252)
above=(P>P.rolling(50,min_periods=35).mean()).mean(axis=1); gate=(above-.5)*2
f=(rel/(rv+1e-8)*(1+.5*gate).to_numpy()[:,None]).shift(1); fr=lp.shift(-10)-lp
A=f.to_numpy(); B=fr.to_numpy(); valid=np.isfinite(A)&np.isfinite(B); n=valid.sum(1); ok=n>=8
AA=np.where(valid,A,np.nan); BB=np.where(valid,B,np.nan); am=np.nanmean(AA,1); bm=np.nanmean(BB,1)
num=np.nansum((AA-am[:,None])*(BB-bm[:,None]),1); den=np.sqrt(np.nansum((AA-am[:,None])**2,1)*np.nansum((BB-bm[:,None])**2,1)); ic=num/den
q=pd.Series(ic,index=f.index)[ok].dropna(); nn=pd.Series(n,index=f.index)[q.index]
print('breadth_conditioned_relative_momentum30 dates',len(q),'avgN',round(nn.mean(),2),'coverage',round(nn.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for z in [120,252,756,1260]:
 x=q.tail(z); print('recent',z,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
for h in [1,3,5,10]:
 BB=(lp.shift(-h)-lp).to_numpy(); vv=np.isfinite(A)&np.isfinite(BB); nn2=vv.sum(1); aa=np.where(vv,A,np.nan);bb=np.where(vv,BB,np.nan); am=np.nanmean(aa,1);bm=np.nanmean(bb,1); num=np.nansum((aa-am[:,None])*(bb-bm[:,None]),1); den=np.sqrt(np.nansum((aa-am[:,None])**2,1)*np.nansum((bb-bm[:,None])**2,1)); x=pd.Series(num/den,index=f.index)[nn2>=8].dropna(); print('horizon',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'obs',len(x))
f.to_csv('scripts/miner_2_20330722_breadth_conditioned_momentum30_signal.csv');q.to_csv('scripts/miner_2_20330722_breadth_conditioned_momentum30_ic.csv')
