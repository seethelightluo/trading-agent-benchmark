import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s,n=4000):
 try: d=get_stock_daily_data(s,n)
 except: d=get_index_daily_data(s,n)
 if d is None or len(d)==0: return None
 d=d.copy(); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
ser={s:get(s) for s in U}; print('available',[(s,v is not None and len(v)) for s,v in ser.items()])
px=pd.DataFrame(ser).sort_index().ffill()
try: v=get_index_daily_data('VIX',4000); v=v.set_index(pd.to_datetime(v.date))['close'].astype(float)
except: v=None
if v is None: v=pd.Series(20.,index=px.index)
v=v.reindex(px.index).ffill()
r=px.pct_change(); mom=px.shift(1).pct_change(20); vol=r.rolling(20).std().shift(1)
vixpct=v.rolling(252,min_periods=100).rank(pct=True).shift(1)
f=mom/vol; f=f.mul(np.where(vixpct.values[:,None]<.5,1,-1),axis=0)
fr=px.pct_change(10).shift(-10); rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append([dt,x[ok].corr(y[ok]),ok.sum()])
ic=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(ic),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean()))
for w in [252,504,756]:
 z=ic.tail(w); print('window',w,'n',len(z),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std())
for h in [1,3,5,10,20]:
 yy=px.pct_change(h).shift(-h); rr=[]
 for dt in f.index:
  a=f.loc[dt]; b=yy.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8: rr.append(a[ok].corr(b[ok]))
 print('decay',h,np.nanmean(rr),len(rr))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.to_csv('scripts/miner_2_20311211_macro_conditioned_trend_signal.csv'); ic.to_csv('scripts/miner_2_20311211_macro_conditioned_trend_ic.csv')
