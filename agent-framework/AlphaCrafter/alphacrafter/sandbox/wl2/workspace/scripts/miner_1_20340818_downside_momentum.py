import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=6000)
    if x is not None and len(x)>300:
        z=x[['date','close']].copy(); z['date']=pd.to_datetime(z.date); z=z.drop_duplicates('date').set_index('date').close.astype(float)
        D[s]=z
p=pd.DataFrame(D).sort_index().ffill(limit=3); r=p.pct_change()
down=r.where(r<0,0).rolling(60).std(); raw=p.pct_change(60)
f=(raw/(down*np.sqrt(60)+1e-8)).shift(1)
for h in [5,10,20,40]:
    fr=p.shift(-h)/p-1; ics=[]; dates=[]; ns=[]
    for dt in f.index:
        a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(a)>=8:
            c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
            if np.isfinite(c): ics.append(c); dates.append(dt); ns.append(len(a))
    q=pd.Series(ics,index=pd.to_datetime(dates)); print('horizon',h,'dates',len(q),'instruments',len(D),'meanIC %.6f ICIR %.6f hit %.3f medN %.1f last %s'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean(),np.median(ns),q.index[-1].date()))
    if h==10:
        rr=f.rank(axis=1,pct=True); print('coverage %.4f turnover %.4f'%(f.notna().mean().mean(),rr.diff().abs().mean(axis=1).dropna().mean()))
        for label,start in [('2020+','2020-01-01'),('2025+','2025-01-01'),('2030+','2030-01-01'),('2033+','2033-01-01')]:
            qq=q[q.index>=start]; print(label,'n',len(qq),'IC %.6f ICIR %.6f'%(qq.mean(),qq.mean()/(qq.std(ddof=1)+1e-12)))
        f.loc[q.index].stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20340818_downside_mom_signal.csv',index=False)
