import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); R=np.log(P).diff(); M=R.mean(axis=1); down=M.where(M<0).fillna(0)
win=60
# pairwise downside covariance, with non-down days represented as zero market shock
mu=down.rolling(win,min_periods=35).mean(); var=((down-mu)**2).rolling(win,min_periods=35).mean()
dbeta=R.sub(R.mean(axis=1),axis=0).mul(down-mu,axis=0).rolling(win,min_periods=35).mean().div(var,axis=0)
downvol=R.where(R<0).rolling(30,min_periods=18).std()
f=(-0.65*dbeta.rank(axis=1,pct=True)-0.35*downvol.rank(axis=1,pct=True)).shift(1)
rows=[]; L=np.log(P); fut={h:L.shift(-h)-L for h in [1,3,5,10,20]}
for dt in f.index:
 a=f.loc[dt]
 for h,y in fut.items():
  b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean())
f.to_csv('scripts/miner_2_20340526_downside_resilience_signal.csv'); z.to_csv('scripts/miner_2_20340526_downside_resilience_ic.csv')
