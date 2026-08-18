import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill()
V=pd.DataFrame({s:d.volume.astype(float) for s,d in D.items()}).reindex(P.index).ffill()
lp=np.log(P); r=lp.diff(); e=r.sub(r.mean(axis=1),axis=0)
# Candidate: volume-confirmed residual trend. Trend is 20-session residual return;
# confirmation is capped log volume surprise versus 60-session median. Lag one day.
trend=e.rolling(20,min_periods=15).sum()
vs=(np.log(V.replace(0,np.nan))-np.log(V.replace(0,np.nan)).rolling(60,min_periods=30).median()).clip(-2,2)
f=(trend*(1+0.25*vs)).rank(axis=1,pct=True).sub(.5,axis=0).shift(1)
rows=[]
for h in [1,3,5,10,20]:
 y=lp.shift(-h)-lp
 for dt in f.index:
  a=f.loc[dt]; b=y.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8 and a[ok].nunique()>1: rows.append((dt,h,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic.dropna(); print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic.dropna()
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean())
f.to_csv('scripts/miner_3_20340721_volume_confirmed_trend_signal.csv'); z.to_csv('scripts/miner_3_20340721_volume_confirmed_trend_ic.csv')
