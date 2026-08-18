import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); V=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).replace(0,np.nan).ffill()
lp=np.log(P); r=lp.diff();
# medium trend relative to cross-section, confirmed by unusually high but non-extreme volume
mom=lp.diff(20); rel=mom-mom.median(axis=1).values[:,None]
rv=r.rolling(20,min_periods=15).std()*np.sqrt(252)
volsur=(np.log(V).diff(5)-np.log(V).diff(40).rolling(20,min_periods=15).mean())
# bounded confirmation avoids one-day volume spikes
confirm=np.tanh(volsur.fillna(0)/2)
f=(rel/(rv+1e-8)*(1+0.35*confirm)).shift(1)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('volume_confirmed_relative_momentum20','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756,1260]:
 x=q.tail(n);print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
for h in [1,3,5,10]:
 ff=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],ff.loc[dt];ok=a.notna()&b.notna()
  if ok.sum()>=8: rr.append(a[ok].corr(b[ok]))
 print('horizon',h,'IC',round(np.nanmean(rr),6),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6),'obs',len(rr))
f.to_csv('scripts/miner_2_20330708_volume_confirmed_momentum20_signal.csv');z.to_csv('scripts/miner_2_20330708_volume_confirmed_momentum20_ic.csv')
