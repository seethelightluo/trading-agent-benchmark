import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().loc[:'2034-05-12'].ffill(); lp=np.log(P); r=lp.diff()
# Volatility-state quality: reward stable positive medium-term trend, penalize recent downside volatility.
trend=r.rolling(40,min_periods=25).sum(); vol=r.rolling(20,min_periods=15).std(); down=r.where(r<0,0).rolling(20,min_periods=15).std()
f=(trend/(vol*np.sqrt(40)+1e-8)) - 0.5*(down/(vol+1e-8))
f=f.rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]
for dt in f.index:
 a=f.loc[dt]; b=(lp.shift(-10)-lp).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=out.ic.dropna()
print('last_date',P.index.max(),'dates',len(q),'avgN',out.n.mean(),'coverage',out.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
for h in [1,5,10,20]:
 yy=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'obs',len(x))
f.to_csv('scripts/miner_2_20340512_stable_trend_signal.csv'); out.to_csv('scripts/miner_2_20340512_stable_trend_ic.csv')
