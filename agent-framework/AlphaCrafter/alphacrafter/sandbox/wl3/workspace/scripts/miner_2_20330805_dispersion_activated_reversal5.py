import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items() for _ in [0]}).ffill(); lp=np.log(P); r=lp.diff()
# High-dispersion short-horizon reversal: reverse 5d relative move, risk scaled,
# with a smooth dispersion activation; lag one session.
raw=lp.diff(5); rel=raw.sub(raw.median(axis=1),axis='index')
vol=r.rolling(20,min_periods=15).std()*np.sqrt(252)
disp=raw.std(axis=1).rolling(60,min_periods=40).rank(pct=True)
activation=(0.5+disp).clip(0.5,1.5)
f=(-rel/(vol+1e-8)).mul(activation,axis=0).shift(1)
for h in [1,3,5,10]:
 fr=lp.shift(-h)-lp; A=f.to_numpy(); B=fr.to_numpy(); valid=np.isfinite(A)&np.isfinite(B); n=valid.sum(1); ok=n>=8
 aa=np.where(valid,A,np.nan);bb=np.where(valid,B,np.nan); am=np.nanmean(aa,1); bm=np.nanmean(bb,1)
 num=np.nansum((aa-am[:,None])*(bb-bm[:,None]),1); den=np.sqrt(np.nansum((aa-am[:,None])**2,1)*np.nansum((bb-bm[:,None])**2,1)); ic=num/den
 q=pd.Series(ic,index=f.index)[ok].dropna(); nn=pd.Series(n,index=f.index)[q.index]
 print('horizon',h,'dates',len(q),'avgN',round(nn.mean(),2),'coverage',round(nn.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
 for z in [120,252,756]:
  x=q.tail(z); print(' recent',z,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
f.to_csv('scripts/miner_2_20330805_dispersion_activated_reversal5_signal.csv')
