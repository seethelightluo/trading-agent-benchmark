import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is None or len(d)==0: d=get_index_daily_data(s,5000)
    if d is not None and len(d):
        x=d.copy(); x['date']=pd.to_datetime(x['date']); frames[s]=x.set_index('date')['close'].astype(float)
px=pd.concat(frames,axis=1).sort_index().ffill()
# Candidate: downside-risk adjusted medium trend, with recent return penalized by downside volatility.
ret=px.pct_change()
down=ret.where(ret<0).rolling(40,min_periods=20).std()*np.sqrt(20)
sig=(px/px.shift(20)-1)/down.replace(0,np.nan)
# winsorize cross-section; signal is observable at t, use forward returns t+1
sig=sig.replace([np.inf,-np.inf],np.nan).shift(1)
fwd=px.pct_change().shift(-1)
rows=[]
for dt in sig.index:
    z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for h in [1,5,10,20]:
    fh=px.pct_change(h).shift(-h)
    rr=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],fh.loc[dt]],axis=1).dropna()
        if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    a=pd.Series(rr).dropna(); print(f'h{h} dates={len(a)} IC={a.mean():.6f} ICIR={a.mean()/a.std():.6f} hit={(a>0).mean():.4f}')
print('assets',len(frames),'dates',len(px),'avgN',r.n.mean(),'coverage',r.n.mean()/len(U),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
print('subperiods')
for y,g in r.groupby(r.index.year):
    if len(g)>50: print(y,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(),(g.ic>0).mean())
out=pd.DataFrame({s:sig[s] for s in U}); out.index.name='date'; out.to_csv('scripts/miner_3_20300822_downside_risk_trend_signal.csv')
