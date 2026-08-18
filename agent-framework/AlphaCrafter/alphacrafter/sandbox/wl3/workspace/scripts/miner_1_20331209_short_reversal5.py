import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Orthogonal short-horizon residual reversal: negate 5-session cross-sectional residual return,
# risk-scaled by trailing 20-session volatility and lagged one session.
raw=lp-lp.shift(5); res=raw.sub(raw.mean(axis=1),axis=0)
vol=r.rolling(20,min_periods=10).std()*np.sqrt(20)
f=(-res.div(vol+1e-12)).shift(1)
f.to_csv('scripts/miner_1_20331209_short_reversal5_signal.csv')
rows=[]
for dt in f.index:
 a,b=f.loc[dt],(lp.shift(-10)-lp).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic.dropna()
print('factor short_residual_reversal5 dates',len(z),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/15,5),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7),'hit',round((q>0).mean(),5),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),5))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
for h in [1,5,10,20]:
 yy=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'obs',len(x))
old=pd.read_csv('scripts/miner_3_20331111_dispersion_scaled_residual_momentum_signal.csv',index_col=0,parse_dates=True)
common=f.index.intersection(old.index); cors=[]
for s in U:
 x=pd.concat([f.loc[common,s],old.loc[common,s]],axis=1).dropna()
 if len(x)>2: cors.append(x.iloc[:,0].corr(x.iloc[:,1]))
print('max_abs_library_correlation_latest',round(float(np.max(np.abs(cors))),7))
z.to_csv('scripts/miner_1_20331209_short_reversal5_ic.csv')
