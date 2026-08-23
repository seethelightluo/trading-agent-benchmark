import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

UNIVERSE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in UNIVERSE:
    df=get_stock_daily_data(symbol=s, days=4000)
    if df is not None and len(df)>150:
        x=df[['date','close','pct_change']].copy(); x['date']=pd.to_datetime(x['date']); x=x.sort_values('date').drop_duplicates('date')
        x['ret']=x['close'].pct_change() if 'pct_change' not in x or x['pct_change'].isna().all() else x['pct_change']/100.0
        frames[s]=x.set_index('date')
print('instruments',len(frames), 'symbols',sorted(frames))
# Union dates and calculate strictly historical signal at t and forward close return t+10.
rows=[]
for s,x in frames.items():
    x=x.copy(); r=x['ret'].astype(float)
    trend=x['close'].pct_change(60)
    persistence=r.rolling(60,min_periods=40).mean().clip(0,1)*2-1
    vol=r.rolling(40,min_periods=25).std()*np.sqrt(40)
    sig=(trend*persistence/vol).replace([np.inf,-np.inf],np.nan)
    fwd=x['close'].shift(-10)/x['close']-1
    z=pd.DataFrame({'date':x.index,'symbol':s,'signal':sig.values,'fwd10':fwd.values}).dropna()
    rows.append(z)
panel=pd.concat(rows,ignore_index=True)
ics=[]
for d,g in panel.groupby('date'):
    if len(g)>=8 and g['signal'].nunique()>1 and g['fwd10'].nunique()>1:
        ics.append((d,g['signal'].corr(g['fwd10']),len(g)))
icdf=pd.DataFrame(ics,columns=['date','ic','n'])
print('period',panel.date.min(),'through',panel.date.max(),'rows',len(panel),'dates',len(icdf),'mean_n',icdf.n.mean(),'coverage',len(panel)/(len(frames)*len(icdf)) if len(icdf) else 0)
mean=icdf.ic.mean(); sd=icdf.ic.std(ddof=1); icir=mean/sd if sd else np.nan
print('10d IC %.9f ICIR %.9f hit %.4f obs %d'%(mean,icir,(icdf.ic>0).mean(),len(icdf)))
for label,a,b in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027-2028','2027-01-01','2028-12-31'),('recent','2028-02-08','2029-02-08')]:
    q=icdf[(icdf.date>=a)&(icdf.date<=b)]
    print(label,len(q), 'IC %.9f ICIR %.9f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1), (q.ic>0).mean()) if len(q)>1 else 'insufficient')
panel.to_csv('scripts/miner_2_20290208_trend_persistence_signal.csv',index=False)
