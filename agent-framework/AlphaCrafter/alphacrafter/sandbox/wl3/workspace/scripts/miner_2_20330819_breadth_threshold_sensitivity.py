import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
rel=lp.diff(30).sub(lp.diff(30).median(axis=1),axis=0); vol=r.rolling(30,min_periods=20).std()*np.sqrt(252)
b=(P>P.rolling(50,min_periods=35).mean()).mean(axis=1)
# Test only threshold sensitivity around the already selected breadth switch.
for lo,hi in [(.40,.60),(.45,.55),(.35,.65),(.50,.50)]:
 reg=pd.Series(np.where(b>=hi,1,np.where(b<=lo,-1,0)),index=P.index)
 f=rel.div(vol+1e-8).mul(reg,axis=0).shift(1)
 h=10; B=(lp.shift(-h)-lp).to_numpy(); A=f.to_numpy(); v=np.isfinite(A)&np.isfinite(B); n=v.sum(1); a=np.where(v,A,np.nan);bb=np.where(v,B,np.nan); am=np.nanmean(a,1); bm=np.nanmean(bb,1); num=np.nansum((a-am[:,None])*(bb-bm[:,None]),1); den=np.sqrt(np.nansum((a-am[:,None])**2,1)*np.nansum((bb-bm[:,None])**2,1)); q=pd.Series(num/den,index=f.index)[n>=8].dropna(); nn=pd.Series(n,index=f.index)[q.index]
 print('thresholds',lo,hi,'dates',len(q),'avgN',round(nn.mean(),2),'coverage',round(nn.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4),'regimes',reg.value_counts(normalize=True).round(3).to_dict())
 for z in [120,252,756]:
  x=q.tail(z); print(' recent',z,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
 if (lo,hi)==(.45,.55): f.to_csv('scripts/miner_2_20330819_breadth_threshold_sensitivity_signal.csv')
