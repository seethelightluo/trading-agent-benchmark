import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Candidate: 20-day signed intraday pressure, normalized by realized range and volume surprise,
# then demeaned cross-section. Only information available at close t is used.
series={}
for s in U:
    df=get_stock_daily_data(s, days=6000)
    if df is None or len(df)<300: df=get_index_daily_data(s, days=6000)
    if df is not None and len(df)>300:
        d=df.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        for c in ['open','close','high','low','volume']:
            d[c]=pd.to_numeric(d[c],errors='coerce')
        rng=(d.high-d.low).replace(0,np.nan)
        pressure=((d.close-d.open)/rng).clip(-1,1)
        volsur=(d.volume/d.volume.rolling(40,min_periods=15).median()).clip(0,5)
        # pressure weighted by abnormal volume; smooth enough for a 10-day rebalance
        raw=(-pressure*volsur).rolling(20,min_periods=15).mean()
        series[s]=raw

panel=pd.DataFrame(series).sort_index()
# dates where all signals and future returns can be measured
closes={}
for s in U:
    df=get_stock_daily_data(s, days=6000)
    if df is None or len(df)<300: df=get_index_daily_data(s, days=6000)
    if df is not None and len(df)>300:
        d=df.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        closes[s]=pd.to_numeric(d.close,errors='coerce')
px=pd.DataFrame(closes).reindex(panel.index)
rets=px.shift(-60)/px-1
for h in [10,20,40,60]:
    fr=px.shift(-h)/px-1
    ics=[]; turns=[]; ninst=[]
    prev=None
    for dt in panel.index:
        x=panel.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            ics.append(z.iloc[:,0].corr(z.iloc[:,1])); ninst.append(len(z))
            ranks=z.iloc[:,0].rank(pct=True)
            if prev is not None:
                turns.append(np.mean((ranks-prev).abs()))
            prev=ranks
    a=np.asarray(ics); a=a[np.isfinite(a)]
    print(f'H={h} dates={len(a)} avg_n={np.mean(ninst):.2f} IC={np.mean(a):.8f} ICIR={np.mean(a)/(np.std(a,ddof=1)/np.sqrt(len(a))):.8f} hit={np.mean(a>0):.4f} turnover={np.mean(turns) if turns else np.nan:.6f}')
# regime breakdown for 60d
fr=px.shift(-60)/px-1
for name,lo,hi in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-35','2033','2035')]:
    vals=[]
    for dt in panel.index:
        if lo<=str(dt.year)<=hi:
            z=pd.concat([panel.loc[dt],fr.loc[dt]],axis=1).dropna()
            if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
    vals=np.asarray(vals); vals=vals[np.isfinite(vals)]
    print('REG',name,'dates',len(vals),'IC',np.mean(vals) if len(vals) else np.nan,'ICIR',np.mean(vals)/(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else np.nan)
print('coverage',panel.notna().mean().mean(),'assets',len(panel.columns),'range',panel.index.min(),panel.index.max())
# artifact, exact signal values for audit
panel.to_csv('scripts/miner_2_20350426_pressure_fade_signal.csv')
