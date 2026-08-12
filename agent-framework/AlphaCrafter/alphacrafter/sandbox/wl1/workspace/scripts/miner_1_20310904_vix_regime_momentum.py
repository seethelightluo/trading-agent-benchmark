import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

SYMS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allx={}
for s in SYMS:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        d=d.copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
        allx[s]=np.log(d['close'].astype(float)).replace([np.inf,-np.inf],np.nan)
px=pd.DataFrame(allx).sort_index()
# observation-only VIX, lagged and aligned
v=get_index_daily_data('VIX', days=5000)
if v is None: v=pd.read_csv('../persistent/index_data/VIX.csv')
v['date']=pd.to_datetime(v['date']); v=v.drop_duplicates('date').set_index('date').sort_index()
vc=(v['close'].astype(float)).reindex(px.index).ffill()
# shock: change over 5d relative to rolling vol, stress indicator bounded
vshock=vc.pct_change(5)/vc.pct_change().rolling(60).std()
# regime gate: trend in calm, reversal during sharp/rising volatility
mom=px-px.shift(20)
sig=mom.mul(np.where(vshock.values[:,None]>0.35,-1.0,1.0),axis=0)
# rank cross section; signal lag means today's factor uses prior completed day
sig=sig.shift(1)
rets={h:px.shift(-h)-px for h in [1,5,10,20]}
rows=[]
for dt in sig.index:
    x=sig.loc[dt]
    n=x.notna().sum()
    if n<8: continue
    row={'date':dt,'n':n}
    for h,r in rets.items():
        y=r.loc[dt]; ok=x.notna()&y.notna()
        row[f'ic{h}']=x[ok].corr(y[ok]) if ok.sum()>=8 else np.nan
    rows.append(row)
z=pd.DataFrame(rows).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',sig.notna().sum().sum()/(len(sig)*len(SYMS)))
for h in [1,5,10,20]:
    a=z[f'ic{h}'].dropna(); print(h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'obs',len(a))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2031')]:
    q=z.loc[a:b,'ic20'].dropna(); print(a+'-'+b,'IC20',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
# turnover rank signal
rank=sig.rank(axis=1,pct=True); turn=rank.diff().abs().mean().mean(); print('turnover_proxy',turn)
print('latest',sig.tail(1).T.to_string())
