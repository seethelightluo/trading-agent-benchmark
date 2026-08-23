import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
    d=get_stock_daily_data(s, days=2400)
    if d is not None and len(d)>100:
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.sort_values('date').drop_duplicates('date'); frames[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(frames).sort_index(); rets=px.pct_change()
# Range-efficiency trend: direction times net displacement / total path, with modest vol normalization.
# Lagged one day; high efficiency means persistent directional movement.
sign=np.sign(px.pct_change(20))
eff=px.pct_change(20).abs()/(rets.abs().rolling(20).sum()+1e-12)
vol=rets.rolling(20).std()*np.sqrt(252)
f=(sign*eff/(vol+1e-12)).shift(1).replace([np.inf,-np.inf],np.nan)
rows=[]; sigrows=[]
for dt in f.index:
    vals=f.loc[dt].dropna();
    if len(vals)>=8:
        sigrows += [{'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)} for s,v in vals.items()]
        for h in [1,3,5,10]:
            future=px.pct_change(h).shift(-h).loc[dt, vals.index].dropna()
            common=vals.index.intersection(future.index)
            if len(common)>=8:
                rows.append((dt,h,len(common),vals.loc[common].corr(future.loc[common])))
r=pd.DataFrame(rows,columns=['date','h','n','ic'])
print('assets',len(frames),'dates',px.index.min(),px.index.max(),'signal_rows',len(sigrows))
for h in [1,3,5,10]:
 x=r[r.h==h].ic.dropna(); print('H',h,'dates',len(x),'avg_n',round(r[r.h==h].n.mean(),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
out=pd.DataFrame(sigrows); out.to_csv('scripts/miner_2_20290222_range_efficiency_trend_signal.csv',index=False)
# coverage and turnover proxy: daily rank signal changes where overlap
print('coverage',round(len(sigrows)/(len(f.index)*len(U)),4))
print('mean_abs_signal',out.signal.abs().mean())
print('signal_dates',out.date.nunique())
