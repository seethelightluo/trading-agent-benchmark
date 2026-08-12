import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

asof='2030-02-07'
acct=get_account_dict(); symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# one candidate: asymmetric trend quality; rewards persistent advances and penalizes frequent downside days
px={}
for s in symbols:
    d=get_stock_daily_data(s, days=2800)
    if d is None or len(d)<150: d=get_index_daily_data(s, days=2800)
    if d is not None and len(d)>0:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d[d.date<=pd.Timestamp(asof)].sort_values('date')
        px[s]=d.set_index('date')['close'].astype(float)
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# asymmetric persistence: medium trend, adjusted by gain/loss-day balance and downside magnitude
trend=close.pct_change(20)
up=ret.gt(0).rolling(30,min_periods=20).mean()
down=ret.lt(0).rolling(30,min_periods=20).mean()
downmag=ret.where(ret<0).abs().rolling(30,min_periods=20).mean()
upmag=ret.where(ret>0).abs().rolling(30,min_periods=20).mean()
# signed trend multiplied by persistence and payoff asymmetry; bounded to reduce outliers
factor=trend * (0.5+up) / (0.5+down) * (1+upmag.fillna(0))/(1+downmag.fillna(0))
factor=factor.replace([np.inf,-np.inf],np.nan)
rows=[]
for h in [1,5,10,20]:
    fwd=close.shift(-h)/close-1
    vals=[]
    for dt in factor.index:
        x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
    q=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
    mean=q.ic.mean(); sd=q.ic.std(ddof=1); icir=mean/sd*np.sqrt(1) if sd else np.nan
    hit=(q.ic>0).mean()
    print(f'H={h} dates={len(q)} avg_n={q.n.mean():.2f} IC={mean:.6f} daily_ICIR={icir:.6f} hit={hit:.4f}')
    for label,a,b in [('2020-2025','2020','2025-12-31'),('2026-2028','2026','2028-12-31'),('2029','2029','2029-12-31'),('2030','2030','2030-02-07')]:
        qq=q[(q.index>=a)&(q.index<=b)]
        if len(qq): print(f'  {label}: dates={len(qq)} IC={qq.ic.mean():.6f} ICIR={qq.ic.mean()/qq.ic.std(ddof=1) if qq.ic.std(ddof=1)>0 else np.nan:.6f}')
# artifact for admission horizon 1 and 10
out=factor.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20300207_asymmetric_persistence_signal.csv',index=False)
print('coverage=',out.symbol.nunique(), 'rows=',len(out), 'date_range=',out.date.min(),out.date.max())
# turnover rank changes
r=factor.rank(axis=1,pct=True); print('turnover=',r.diff().abs().mean().mean())
