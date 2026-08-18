import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).sort_index().ffill(); lp=np.log(P); r=lp.diff()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index(); vc='close' if 'close' in v else list(v.columns)[0]
vv=pd.to_numeric(v[vc],errors='coerce').reindex(P.index).ffill(); stress=(vv/vv.rolling(60,min_periods=40).median()-1).clip(0,1)
base=-(lp-lp.shift(5)); f=(base.rank(axis=1,pct=True)-.5).mul(1+stress,axis=0).shift(1)
rows=[]
for h in [1,3,5,10,20]:
 y=lp.shift(-h)-lp
 for i,dt in enumerate(f.index):
  a=f.iloc[i].values; b=y.iloc[i].values; ok=np.isfinite(a)&np.isfinite(b)
  if ok.sum()>=8 and np.unique(a[ok]).size>1: rows.append((dt,h,np.corrcoef(a[ok],b[ok])[0,1],ok.sum()))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,3,5,10,20]:
 q=z[z.h==h].ic; print('horizon',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
q=z[z.h==10].set_index('date').ic
for n in [120,252,756,1260]:
 x=q.tail(n); print('recent',n,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
print('turn',f.rank(pct=True).diff().abs().mean(axis=1).mean(),'coverage',f.notna().mean().mean(),'dates',len(q),'avg instruments',z[z.h==10].n.mean(),'vixvalid',vv.notna().mean())
f.to_csv('scripts/miner_3_20340915_stress_reversal_signal.csv'); z.to_csv('scripts/miner_3_20340915_stress_reversal_ic.csv')
