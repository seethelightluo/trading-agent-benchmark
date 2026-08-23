import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is None or len(d)<100: d=get_index_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        frames[s]=d['close'].astype(float)
px=pd.DataFrame(frames).sort_index().ffill()
# avoid lookahead: signal on t predicts return t+1..t+h
ret=px.pct_change()
down=ret.clip(upper=0).rolling(40,min_periods=30).std()*np.sqrt(40)
# downside-adjusted medium-term continuation, interpretable
sig=px.pct_change(20)/down.replace(0,np.nan)
# cross-sectional daily IC, horizons
print('range',px.index.min(),px.index.max(),'dates',len(px),'assets',len(px.columns))
for h in [5,10,20]:
    fwd=px.shift(-h)/px-1
    vals=[]; dates=[]; counts=[]
    for dt in px.index:
        x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); counts.append(len(z))
    a=pd.Series(vals,index=dates).dropna(); ic=a.mean(); sd=a.std(ddof=1); icir=ic/sd*np.sqrt(252) if sd else np.nan
    print('H',h,'n_dates',len(a),'avg_n',np.mean(counts),'coverage',np.mean(counts)/15,'IC',round(ic,6),'ICIR',round(icir,4),'hit',round((a>0).mean(),4))
# turnover proxy rank changes over 10-day decisions
r=sig.rank(axis=1,pct=True); turnover=(r-r.shift(10)).abs().mean(axis=1).dropna().mean()
print('turnover10',round(float(turnover),6),'valid_signal_coverage',round(float(sig.notna().sum(axis=1).div(15).mean()),6))
# regimes for 10d
fwd=px.shift(-10)/px-1; vals=[]
for dt in px.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
a=pd.Series(dict(vals)).dropna()
for yr in sorted(set(a.index.year)):
 q=a[a.index.year==yr]
 if len(q)>20: print('REG',yr,len(q),round(q.mean(),5))
