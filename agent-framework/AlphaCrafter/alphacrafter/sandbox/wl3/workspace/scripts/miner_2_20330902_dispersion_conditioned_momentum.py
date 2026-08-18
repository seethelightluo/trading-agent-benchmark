import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); L=np.log(P); R=L.diff()
# Dispersion-conditioned relative momentum: trend in quiet cross-sections, reversal after unusually dispersed moves.
base=L.diff(30).sub(L.diff(30).median(axis=1),axis=0)
vol=R.rolling(30,min_periods=20).std()*np.sqrt(252)
disp=R.rolling(5,min_periods=4).std().mean(axis=1)
z=(disp-disp.rolling(120,min_periods=60).mean())/(disp.rolling(120,min_periods=60).std()+1e-12)
state=pd.Series(np.where(z>=0, -1, 1),index=P.index)
f=base.div(vol+1e-8).mul(state,axis=0).shift(1)
h=10; B=(L.shift(-h)-L).to_numpy(); A=f.to_numpy(); v=np.isfinite(A)&np.isfinite(B); n=v.sum(1)
a=np.where(v,A,np.nan); bb=np.where(v,B,np.nan); am=np.nanmean(a,1); bm=np.nanmean(bb,1)
num=np.nansum((a-am[:,None])*(bb-bm[:,None]),1); den=np.sqrt(np.nansum((a-am[:,None])**2,1)*np.nansum((bb-bm[:,None])**2,1))
q=pd.Series(num/den,index=f.index)[n>=8].dropna(); nn=pd.Series(n,index=f.index)[q.index]
print('dates',len(q),'avgN',round(nn.mean(),2),'coverage',round(nn.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for z0 in [120,252,756]:
 x=q.tail(z0); print('recent',z0,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('state share',state.value_counts(normalize=True).round(3).to_dict())
f.to_csv('scripts/miner_2_20330902_dispersion_conditioned_momentum_signal.csv')
q.to_csv('scripts/miner_2_20330902_dispersion_conditioned_momentum_ic.csv',header=['ic'])
