import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=4000)
    if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
    if d is not None and len(d)>0:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill(limit=3)
r=np.log(px).diff()
# volatility compression: recent 10d vol relative to 60d vol, inverted; lagged one day
v10=r.rolling(10).std(); v60=r.rolling(60).std()
f=-(v10/v60).shift(1)
# winsorize cross-section not needed; evaluate rank IC
for h in [5,10,20]:
    fr=np.log(px).shift(-h)-np.log(px)
    rows=[]; cov=[]
    for dt in f.index:
        a=f.loc[dt]; b=fr.loc[dt]
        z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); cov.append(len(z)/15)
    x=pd.Series(rows).dropna();
    print('H',h,'dates',len(x),'avgN',np.mean(np.array(cov)*15) if cov else 0,'coverage',np.mean(cov) if cov else 0,'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1)*np.sqrt(252) if len(x)>1 else np.nan,'hit',np.mean(x>0))
# turnover daily rank signal
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('turnover_proxy',turn,'assets',len(px.columns),'dates',len(px),'start',px.index.min(),'end',px.index.max())
# save signal provenance
out=f.copy(); out.to_csv('scripts/miner_1_20290111_vol_compression_signal.csv')
