import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s,5000)
    if d is None or len(d)<80: d=get_index_daily_data(s,5000)
    if d is not None:
        d=d[['date','close']].copy(); d['date']=pd.to_datetime(d.date); d=d.drop_duplicates('date').set_index('date').close
        frames[s]=d
px=pd.DataFrame(frames).sort_index().ffill()
# Candidate: lagged 20-session risk-adjusted momentum, cross-sectionally ranked.
ret=px.pct_change(20); vol=px.pct_change().rolling(20).std(); sig=(ret/vol).shift(1)
# forward 10d return, aligned strictly after signal date
fwd=px.shift(-10)/px-1
obs=[]; turnovers=[]; counts=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        obs.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))); counts.append(len(z))
    prev=sig.shift(1).loc[dt]
    a=x.dropna().rank(pct=True); b=prev.reindex(a.index).dropna().rank(pct=True)
    if len(a)>0 and len(b)>0: turnovers.append(np.mean(abs(a.reindex(b.index)-b)))
ic=pd.Series(dict(obs)).dropna();
print('dates',len(ic),'avg_n',np.mean(counts),'coverage',np.mean(counts)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turnovers))
for h in [1,5,10,20]:
    yy=px.shift(-h)/px-1; oo=[]
    for dt in sig.index:
        z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
        if len(z)>=8: oo.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
    print('decay',h,np.nanmean(oo),len(oo))
for n in [365,750,1260]:
    q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
# signal artifact for deterministic audit
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20350607_volscaled_momentum_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_3_20350607_volscaled_momentum_ic.csv',index=False)
