import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
asof=pd.Timestamp('2030-03-20')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in syms:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d): p[s]=d[d.date<=asof].set_index('date').close.astype(float)
c=pd.DataFrame(p).reindex(columns=syms); r=c.pct_change()
# Stress-conditioned cross-asset reversal: recent losers are favored more when VIX is elevated.
vix=pd.read_csv('../persistent/index_data/VIX.csv')
vix['date']=pd.to_datetime(vix['date']); vcol='close' if 'close' in vix else [x for x in vix.columns if x not in ['date']][0]
v=vix.set_index('date')[vcol].astype(float).reindex(c.index).ffill()
# lag all observable inputs by one completed session
stress=(v.rolling(60,min_periods=30).mean()-v.rolling(252,min_periods=80).mean())/(v.rolling(252,min_periods=80).std()+1e-9)
stress=stress.clip(-2,2)
f=(-r.rolling(5).sum()/(r.rolling(20).std()*np.sqrt(20)+0.02)).mul(1+0.45*stress,axis=0).shift(1)
rows=[]
for dt in f.index:
 for h in [1,5,10,20]:
  z=pd.concat([f.loc[dt],(c.shift(-h)/c-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,h,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
q=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10,20]:
 x=q[q.h==h].set_index('date'); print('H',h,'dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),6),'hit',round((x.ic>0).mean(),4))
 for a,b in [('2020','2025-12-31'),('2026','2028-12-31'),('2029','2029-12-31'),('2030','2030-03-20')]:
  y=x[(x.index>=a)&(x.index<=b)]
  if len(y)>2: print(' ',a,len(y),round(y.ic.mean(),6),round(y.ic.mean()/y.ic.std(ddof=1),6))
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20300320_stress_reversal_signal.csv',index=False)
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'coverage',round(out.symbol.nunique()/15,4),'rows',len(out),'macro_obs',v.notna().sum())
