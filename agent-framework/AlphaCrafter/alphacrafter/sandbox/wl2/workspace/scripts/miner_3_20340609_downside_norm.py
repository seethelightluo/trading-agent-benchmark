import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:fetch(s) for s in U}; p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items() if x is not None},axis=1).sort_index(); r=np.log(p).diff()
loc=(p-p.rolling(120,min_periods=60).min())/(p.rolling(120,min_periods=60).max()-p.rolling(120,min_periods=60).min()+1e-12)
down=r.where(r<0).rolling(20,min_periods=5).std(); total=r.rolling(60,min_periods=20).std(); s=(1-loc)/(1+down/(total+1e-12)); f=s.rank(axis=1,pct=True).shift(1)
rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],(p.shift(-40)/p-1).loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-06-07']
print('factor=downside-risk normalized 120d range reversal; dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-06-07'])]: print(lab,len(qq),qq.ic.mean(),qq.ic.mean()/qq.ic.std(ddof=1),(qq.ic>0).mean())
for h in [10,20,40]:
 rr=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rr.append(a.iloc[:,0].corr(a.iloc[:,1]))
 print('decay',h,np.nanmean(rr),len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_3_20340609_downside_norm_signal.csv'); z.reset_index().to_csv('scripts/miner_3_20340609_downside_norm_ic.csv',index=False)
