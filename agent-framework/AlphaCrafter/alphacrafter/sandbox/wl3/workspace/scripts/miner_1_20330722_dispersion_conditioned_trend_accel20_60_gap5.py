import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
# Trend acceleration: short trend in excess of linearized medium trend, risk scaled.
acc=(lp.diff(20)-lp.diff(60)/3.0)
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
base=acc/(vol+1e-8)
# Cross-asset realized dispersion regime, known only at signal timestamp and lagged with signal.
disp=r.rolling(20,min_periods=15).std().mean(axis=1)
med=disp.rolling(252,min_periods=60).median()
# Allow trend signal full strength in high-dispersion regimes, half strength otherwise.
gate=pd.Series(np.where(disp>med,1.0,0.5),index=disp.index)
f=base.mul(gate,axis=0).shift(5).shift(1)
rows=[]
for dt in f.index:
 a,b=f.loc[dt],(lp.shift(-10)-lp).loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.ic
print('dispersion_conditioned_trend_accel20_60_gap5','dates',len(z),'avgN',round(z.n.mean(),3),'coverage',round(z.n.mean()/15,5),'IC',round(q.mean(),7),'ICIR',round(q.mean()/q.std(ddof=1),7),'hit',round((q>0).mean(),5),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),5))
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),5))
for h in [1,5,10,20]:
 y=lp.shift(-h)-lp; rr=[]
 for dt in f.index:
  a,b=f.loc[dt],y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rr.append(a[ok].corr(b[ok]))
 x=pd.Series(rr).dropna(); print('horizon',h,'IC',round(x.mean(),7),'ICIR',round(x.mean()/x.std(ddof=1),6),'obs',len(x))
f.to_csv('scripts/miner_1_20330722_dispersion_conditioned_trend_accel20_60_gap5_signal.csv'); z.to_csv('scripts/miner_1_20330722_dispersion_conditioned_trend_accel20_60_gap5_ic.csv')
