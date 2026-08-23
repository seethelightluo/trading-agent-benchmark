import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
acct=get_account_dict(); wl=acct.get('watch_list') or watch
# Explicit benchmark universe, avoid observation-only macro series
wl=[x for x in watch if x in (wl or watch)] if wl else watch
prices={}
for s in wl:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d): prices[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(prices).sort_index().ffill()
ret=px.pct_change()
# factor known at date t: short reversal normalized by trailing vol, active in elevated dispersion
csdisp=ret.std(axis=1, ddof=0)
disp_med=csdisp.rolling(60,min_periods=40).median().shift(1)
active=(csdisp.shift(1)>disp_med).astype(float)
vol=ret.rolling(20,min_periods=15).std().shift(1)*np.sqrt(20)
shortret=px.pct_change(5).shift(1)
f=(-shortret/vol).mul(active,axis=0)
# require enough observations and evaluate forward 10d return from t close
fwd=px.shift(-10)/px-1
rows=[]; ics=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]
    ok=a.notna()&b.notna()
    if ok.sum()>=8:
        ic=a[ok].corr(b[ok],method='spearman')
        if pd.notna(ic): ics.append((dt,ic,ok.sum()))
# turnover as rank changes on consecutive valid dates
sig=f.rank(axis=1,pct=True)
turn=(sig.diff().abs().mean(axis=1)).dropna().mean()
icdf=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date')
mean=icdf.ic.mean(); sd=icdf.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
print('dates',len(icdf),'start',icdf.index.min(),'end',icdf.index.max(),'avg_n',icdf.n.mean())
print('coverage',float(f.notna().mean().mean()),'mean_ic',mean,'icir',icir,'hit',float((icdf.ic>0).mean()),'turnover',turn)
for h in [1,5,10,20]:
    fw=px.shift(-h)/px-1; vals=[]
    for dt in f.index:
        ok=f.loc[dt].notna()&fw.loc[dt].notna()
        if ok.sum()>=8:
            z=f.loc[dt,ok].corr(fw.loc[dt,ok],method='spearman')
            if pd.notna(z): vals.append(z)
    print('decay',h,np.mean(vals),len(vals))
# recent regimes
for label, sub in [('365d',icdf.tail(252)),('180d',icdf.tail(126)),('2032YTD',icdf[icdf.index>='2032-01-01'])]:
    if len(sub)>5: print(label,'n',len(sub),'ic',sub.ic.mean(),'icir',sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252))
# artifacts
out=f.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_1_20320429_stress_dispersion_reversal_signal.csv',index=False)
icdf.to_csv('scripts/miner_1_20320429_stress_dispersion_reversal_ic.csv')
