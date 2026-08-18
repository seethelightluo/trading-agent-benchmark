import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P); r=lp.diff()
formation=r.shift(20).rolling(60).sum(); vol=r.shift(20).rolling(40).std(); med=formation.median(axis=1)
f=formation.sub(med,axis=0).div(vol.replace(0,np.nan)).mul(-1); f=f.loc[:'2033-06-10']
fwd=lp.shift(-10)-lp
rows=[]; prev=None; turns=[]
for dt in f.index:
 a,b=f.loc[dt],fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
  rr=a[ok].rank(pct=True)
  if prev is not None: turns.append((rr-prev).abs().mean())
  prev=rr
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,5),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(np.mean(turns),5))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
print('period',z.index.min(),z.index.max())
f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20330610_inverse_relative_mr60_gap20_signal.csv',index=False)
z.reset_index().to_csv('scripts/miner_1_20330610_inverse_relative_mr60_gap20_ic.csv',index=False)
