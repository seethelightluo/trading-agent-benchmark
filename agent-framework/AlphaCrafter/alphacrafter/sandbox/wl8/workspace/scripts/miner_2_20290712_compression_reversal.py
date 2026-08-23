import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 try:d=get_stock_daily_data(s,days=4000)
 except:continue
 if d is None or len(d)<250:continue
 d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date'); r=np.log(d.close).diff(); rng=(d.high-d.low).replace(0,np.nan)
 # lagged reversal, scaled up after compressed ranges (mean reversion after quiet regime)
 comp=(rng.rolling(5).mean()/rng.rolling(20).mean()).shift(1)
 f=(-r.rolling(5).sum().shift(1)/r.rolling(20).std().shift(1))*((1-comp).clip(-1,1))
 d['f']=f.replace([np.inf,-np.inf],np.nan)
 for h in [1,3,5,10]:d['y'+str(h)]=np.log(d.close).shift(-h)-np.log(d.close)
 fs[s]=d.reset_index()[['date','f','y1','y3','y5','y10']]
x=pd.concat([z.assign(s=s) for s,z in fs.items()]); out=[]
for dt,g in x.groupby('date'):
 for h in [1,3,5,10]:
  q=g[['f','y'+str(h)]].dropna()
  if len(q)>=8:out.append((dt,len(q),h,q.f.corr(q['y'+str(h)])))
o=pd.DataFrame(out,columns=['date','n','h','ic']);print('dates',len(o),'avg_n',o.n.mean(),'coverage',x.f.notna().groupby(x.date).mean().mean())
for h in [1,3,5,10]:
 q=o[o.h==h].ic;print(h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'recent180',q.tail(180).mean(),q.tail(180).mean()/q.tail(180).std(ddof=1),'recent360',q.tail(360).mean(),q.tail(360).mean()/q.tail(360).std(ddof=1))
x.to_csv('scripts/miner_2_20290712_compression_reversal_signal.csv',index=False);print('rows',len(x))
