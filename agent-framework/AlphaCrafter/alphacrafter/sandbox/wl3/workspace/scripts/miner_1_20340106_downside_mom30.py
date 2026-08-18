import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Downside-adjusted medium momentum: 30-session log return divided by downside deviation,
# rewarding assets with positive trends and penalizing realized negative-day risk.
down=r.where(r<0,0.0).pow(2).rolling(30,min_periods=15).mean().pow(.5)*np.sqrt(30)
f=(lp-lp.shift(30)).div(down+1e-12).sub(((lp-lp.shift(30)).div(down+1e-12)).mean(axis=1),axis=0).shift(1)
rows=[]
for dt in f.index:
 a,b=f.loc[dt],(lp.shift(-10)-lp).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
print('downside_mom30 dates',len(z),'avgN',z.n.mean(),'coverage',z.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turn',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'ICIR',x.mean()/x.std(ddof=1),'IC',x.mean(),'hit',(x>0).mean())
for h in [5,10,20]:
 yy=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'obs',len(x))
z.to_csv('scripts/miner_1_20340106_downside_mom30_ic.csv'); f.to_csv('scripts/miner_1_20340106_downside_mom30_signal.csv')
