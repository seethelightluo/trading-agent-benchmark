import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        x=d[['date','close']].copy(); x['date']=pd.to_datetime(x.date); raw[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(raw).sort_index().ffill(limit=3); r=np.log(p).diff(); common=r.median(axis=1); res=r.sub(common,axis=0)
f=-(res.rolling(5).sum()/res.rolling(20).std()).replace([np.inf,-np.inf],np.nan)
out=[]
for h in [1,5,10,20]:
 fr=np.log(p).shift(-h)-np.log(p); vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); out.append((h,len(a),a.mean(),a.std(ddof=1),a.mean()/a.std(ddof=1)*np.sqrt(len(a)),(a>0).mean()))
ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna().mean(); cov=f.notna().sum(axis=1).mean()/len(U)
print('dates',len(p),'instruments',len(raw),'range',p.index.min(),p.index.max()); print('coverage',round(cov,4),'turnover',round(turnover,4))
for x in out: print('H%d n=%d IC=%.8f ICIR=%.8f hit=%.4f'%(x[0],x[1],x[2],x[4],x[5]))
fr=np.log(p).shift(-1)-np.log(p); vals=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals)).dropna(); print('halves',a[a.index<p.index[int(len(p)/2)]].mean(),a[a.index>=p.index[int(len(p)/2)]].mean())
f.reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20341002_residual_reversal_signal.csv',index=False)
