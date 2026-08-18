import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
px=pd.DataFrame(D).sort_index(); r=px.pct_change()
# one idea: downside-risk normalized 5d reversal, with downside semideviation over 20 sessions
fac=-(px/px.shift(5)-1)/(r.where(r<0).rolling(20,min_periods=10).std()*np.sqrt(20))
# forward returns at several horizons, daily cross-sectional rank IC
for h in [1,5,10]:
    fwd=px.shift(-h)/px-1; vals=[]; ns=[]; dates=[]
    for dt in fac.index:
        x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
    a=np.array(vals); print(h,'obs',len(a),'meanN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean(ns)/15)
# turnover rank changes
ranks=fac.rank(axis=1,pct=True); ch=(ranks.diff().abs().mean(axis=1)>0.08).mean(); print('turnover_proxy',ch)
# latest valid
print('latest',fac.dropna(how='all').iloc[-1].to_dict())
# pooled corr with simple library proxies
proxies={'rev5':-(px/px.shift(5)-1),'mom20':(px/px.shift(20)-1)/r.rolling(20).std(),'lead':(px/px.shift(5)-1)-((px/px.shift(5)-1).median(axis=1).values[:,None])}
for n,v in proxies.items(): print('corr',n,fac.stack().corr(v.stack()))
