import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,4000) for s in U}; v=get_index_daily_data('VIX',4000)
for s in U:
    px[s]['date']=pd.to_datetime(px[s].date); px[s]=px[s].set_index('date')['close']
v['date']=pd.to_datetime(v.date); v=v.set_index('date')['close']
dates=sorted(set.intersection(*[set(x.index) for x in px.values()]))
rows=[]
for d in dates:
    vv=v.loc[:d].dropna()
    if len(vv)<61: continue
    # continuous stress-transition weight: current VIX percentile versus trailing 60d,
    # with extra emphasis on rising stress; entirely lagged at decision date.
    med=vv.iloc[-61:-1].median(); q=vv.iloc[-61:-1].rank(pct=True).iloc[-1] if len(vv)>=61 else np.nan
    stress=np.clip((vv.iloc[-1]/(med+1e-12)-1)*4, -1, 2)
    rising=np.clip((vv.iloc[-1]/(vv.iloc[-4]+1e-12)-1)*8, -1, 1)
    regime=0.25+np.clip(stress,0,1)+0.35*np.clip(rising,0,1)
    r={s:px[s].loc[:d].pct_change(5).iloc[-1] for s in U}
    vol={s:px[s].loc[:d].pct_change().iloc[-21:-1].std() for s in U}
    z=pd.Series(r); sig=-(z-z.median())/pd.Series(vol).replace(0,np.nan)*regime
    for s in U:
        fut=px[s].loc[px[s].index>d]
        if pd.notna(sig[s]) and len(fut)>=10:
            for h in [1,3,5,10]: rows.append((d,s,float(sig[s]),h,float(fut.iloc[h-1]/px[s].loc[d]-1)))
df=pd.DataFrame(rows,columns=['date','s','sig','h','fut'])
df.to_csv('scripts/miner_2_20330624_continuous_stress_transition_signal.csv',index=False)
print('dates',len(dates),'used_dates',df.date.nunique(),'rows',len(df),'avg_n',df.groupby('date').size().mean(),'coverage',len(df)/(len(dates)*15*4))
for h,g in df.groupby('h'):
    ic=g.groupby('date').apply(lambda x:x.sig.corr(x.fut),include_groups=False).dropna()
    print('H',h,'IC',round(g.sig.corr(g.fut),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'n_dates',len(ic))
    for a,b in [('2020','2025'),('2026','2029'),('2030','2033')]:
        z=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]
        print(a+'-'+b,'n',len(z),'ic',round(z.mean(),6) if len(z) else None,'ir',round(z.mean()/z.std(ddof=1),5) if len(z)>1 else None)
