import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',usecols=['date','close']);d.date=pd.to_datetime(d.date);return d.set_index('date').close.sort_index()
p=pd.concat({s:ld(s) for s in U},axis=1);p=p.loc[:,~p.columns.duplicated()];r=np.log(p).diff();ret5=r.rolling(5).sum();vol20=r.rolling(20,min_periods=15).std();disp=ret5.sub(ret5.median(axis=1),axis=0).abs().median(axis=1);scale=(1+disp/(disp.rolling(120,min_periods=60).median()+1e-12)).clip(.5,2.);f=(-ret5/(vol20*np.sqrt(5))*scale.values[:,None]).shift(1)
print('assets',p.shape[1],'dates',len(r),'avg_n',f.notna().sum(axis=1).mean(),'coverage',f.notna().sum(axis=1).mean()/p.shape[1])
for h in [1,3,5,10]:
 y=r.rolling(h).sum().shift(-h);vals=[]
 for a,b in zip(f.to_numpy(),y.to_numpy()):
  ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8:
   rx=pd.Series(a[ok]).rank().to_numpy();rz=pd.Series(b[ok]).rank().to_numpy();vals.append(np.corrcoef(rx,rz)[0,1])
 q=np.asarray(vals);ic=np.nanmean(q);print('H',h,'dates',len(q),'IC',ic,'ICIR',ic/np.nanstd(q,ddof=1),'hit',(q>0).mean())
f.to_csv('scripts/miner_1_20330513_smooth_dispersion_reversal_signal.csv')
