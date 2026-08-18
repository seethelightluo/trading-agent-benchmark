import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill(); lp=np.log(P)
# Candidate: medium-term momentum with a 10-session information gap and volatility scaling.
# Signal at t uses data through t-10; target is t+1..t+10 return.
r60=lp.diff(60); vol20=lp.diff().rolling(20).std()*np.sqrt(252)
f=(r60/vol20).shift(10)
fr=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('volscaled_momentum60_gap10','dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'ICIR',round(x.mean()/x.std(ddof=1),5),'IC',round(x.mean(),5),'hit',round((x>0).mean(),4))
for h in [1,3,5,10]:
 rr=lp.shift(-h)-lp; rows2=[]
 for dt in f.index:
  a,b=f.loc[dt],rr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rows2.append(a[ok].corr(b[ok]))
 x=pd.Series(rows2).dropna(); print('decay',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6))
f.to_csv('scripts/miner_1_20330708_volscaled_momentum60_gap10_signal.csv'); z.to_csv('scripts/miner_1_20330708_volscaled_momentum60_gap10_ic.csv')
