import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(P.index).ffill()
# Stress-conditioned cross-asset short-term reversal: reversal is emphasized when VIX is unusually elevated.
z=(vix-vix.rolling(120,min_periods=60).mean())/vix.rolling(120,min_periods=60).std()
stress=(1+z.clip(lower=0,upper=2)/2)
f=(-r.rolling(5,min_periods=5).sum()).mul(stress,axis=0).shift(1)
y=lp.shift(-10)-lp
rows=[]
for dt in f.index:
 a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').ic
print('stress_conditioned_reversal5','dates',len(q),'avgN',round(pd.DataFrame(rows,columns=['d','i','n']).n.mean(),3),'coverage',round(pd.DataFrame(rows,columns=['d','i','n']).n.mean()/15,5),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7),'hit',round((q>0).mean(),5),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),5))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
for h in [1,5,10,20]:
 yy=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'obs',len(x))
f.to_csv('scripts/miner_1_20330902_stress_conditioned_reversal5_signal.csv'); q.to_csv('scripts/miner_1_20330902_stress_conditioned_reversal5_ic.csv')
