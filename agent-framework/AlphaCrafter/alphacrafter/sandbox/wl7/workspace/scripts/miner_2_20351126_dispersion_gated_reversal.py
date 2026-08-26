import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
    try: x=get_index_daily_data(s,5000)
    except Exception: x=None
    if x is None or len(x)<100:
        try: x=get_stock_daily_data(s,5000)
        except Exception: x=None
    if x is None:return None
    x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date')['close'].astype(float)
    return x.replace([np.inf,-np.inf],np.nan)
P={s:load(s) for s in U}; P={s:x for s,x in P.items() if x is not None}
px=pd.DataFrame(P).sort_index().ffill(limit=3); r=np.log(px).diff()
ret5=r.rolling(5).sum(); vol20=r.rolling(20).std().clip(lower=1e-6); disp=r.rolling(20).std().mean(axis=1)
zdisp=(disp-disp.rolling(120).mean())/disp.rolling(120).std().clip(lower=1e-8)
gate=(1/(1+np.exp(-zdisp))).clip(.15,.85); f=(-ret5/vol20).sub((-ret5/vol20).mean(axis=1),axis=0).mul(gate,axis=0).shift(1)
fr=px.shift(-20)/px-1; rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna(); mean=x.ic.mean(); sd=x.ic.std(ddof=1)
print('factor=dispersion_gated_relative_reversal5_h20'); print('dates',len(x),'avg_n',x.n.mean(),'coverage',f.notna().sum().sum()/f.size,'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(x.ic>0).mean())
for lo,hi in [('2020','2024'),('2025','2029'),('2030','2034'),('2035','2035')]:
 y=x.loc[lo:hi].ic; print(lo,'n',len(y),'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1)*np.sqrt(252) if len(y)>2 else np.nan)
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20351126_dispersion_gated_reversal_signal.csv',index=False); print('rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
