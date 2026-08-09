import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
    for f in (get_index_daily_data,get_stock_daily_data):
        try:
            x=f(s, days=2200)
            if x is not None and len(x)>100: return x[['date','close']].copy()
        except Exception: pass
    return None
xs={s:fetch(s) for s in U}; xs={s:x for s,x in xs.items() if x is not None}
print('instruments',len(xs),sorted(xs))
px=pd.concat([x.assign(symbol=s) for s,x in xs.items()]).pivot(index='date',columns='symbol',values='close').sort_index().ffill()
# volatility-adjusted 20d momentum: prior 20d return divided by trailing 20d realized vol, lagged one day
r=px.pct_change(); mom=px.pct_change(20).shift(1); vol=r.rolling(20).std().shift(1)*np.sqrt(20)
f=mom/vol
# forward horizons, date-wise Spearman IC, require >=8 names
for h in [1,3,5,10]:
    fr=px.shift(-h)/px-1
    vals=[]; nms=[]
    for d in f.index:
        z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
        if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); nms.append(len(z))
    a=np.array(vals); print('H',h,'dates',len(a),'avgN',np.mean(nms),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'med',np.nanmedian(a))
# coverage and turnover rank
valid=f.notna().sum(axis=1)/len(U); ranks=f.rank(axis=1,pct=True)
to=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna()
print('coverage',valid.mean(),'turnover',to.mean(),'period',px.index.min(),px.index.max())
# annual IC 5d
fr=px.shift(-5)/px-1; zics=[]
for d in f.index:
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:zics.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
q=pd.DataFrame(zics,columns=['date','ic']); print(q.groupby(q.date.dt.year).ic.agg(['count','mean']))
