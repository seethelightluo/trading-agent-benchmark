import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_index_daily_data, get_stock_daily_data

acct=get_account_dict(); syms=acct.get('watch_list',[])
if not syms: syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
prices={}
for s in syms:
    d=get_stock_daily_data(s, 5000)
    if d is None or len(d)<100: d=get_index_daily_data(s,5000)
    if d is not None and len(d): prices[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
px=pd.DataFrame(prices).sort_index().ffill()
# skip-5 medium reversal: negative return over t-25 to t-5, scaled by 40d realized vol
ret=px.pct_change()
vol=ret.rolling(40,min_periods=30).std()*np.sqrt(20)
factor=-(px.shift(5)/px.shift(25)-1)/vol
# lagged availability is naturally signal at t, forward begins t+1
for h in [1,5,10,20,30]:
    fr=px.shift(-h)/px-1
    vals=[]; dates=[]
    for dt in factor.index:
        x=factor.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
    a=np.asarray(vals,float); a=a[np.isfinite(a)]
    ic=a.mean(); ir=ic/a.std(ddof=1) if len(a)>1 else np.nan
    thirds=[np.mean(q) for q in np.array_split(a,3)]
    print(f'H{h} IC {ic:.6f} ICIR {ir:.6f} hit {np.mean(a>0):.4f} dates {len(a)} thirds {[round(x,6) for x in thirds]}')
valid=factor.notna().sum(axis=1); print('assets',len(syms),'dates',len(px),'avg coverage',round((valid/len(syms)).mean(),4),'latest',px.index[-1].date())
# average daily rank signal turnover proxy
r=factor.rank(axis=1,pct=True); print('turnover',round(r.diff().abs().mean(axis=1).mean(),4))
# save artifact
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20330516_skip5_medium_reversal_signal.csv',index=False)
