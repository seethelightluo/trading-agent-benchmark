import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(s):
    for fn in (get_stock_daily_data,get_index_daily_data):
        try:
            x=fn(s, days=5000)
            if x is not None and len(x): return x.copy()
        except Exception: pass
    return None
rows=[]
for s in U:
    x=load(s)
    if x is None: continue
    x['date']=pd.to_datetime(x['date']); x=x.sort_values('date')
    x['overnight']=x['open']/x['close'].shift(1)-1
    x['fwd1']=x['close'].shift(-1)/x['close']-1
    # contrarian overnight shock, mild winsorization cross-section later
    x['signal']=-x['overnight'].clip(-.15,.15)
    x['symbol']=s
    rows.append(x[['date','symbol','signal','fwd1']])
d=pd.concat(rows,ignore_index=True).dropna()
d=d[d.date>=pd.Timestamp('2020-01-01')]
ics=[]; artifact=[]
for dt,g in d.groupby('date'):
    if len(g)>=8 and g.signal.nunique()>1 and g.fwd1.nunique()>1:
        ic=g.signal.corr(g.fwd1,method='spearman')
        if pd.notna(ic): ics.append(ic)
    for _,r in g.iterrows(): artifact.append({'date':dt.strftime('%Y-%m-%d'),'symbol':r.symbol,'signal':float(r.signal)})
ics=np.array(ics)
mean=ics.mean(); sd=ics.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
hit=(ics>0).mean()
# turnover rank changes across sequential dates
p=d.pivot(index='date',columns='symbol',values='signal').sort_index(); ranks=p.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).mean()
print(json.dumps({'dates':len(ics),'rows':len(d),'avg_instruments':len(d)/d.date.nunique(),'coverage':len(d)/(d.date.nunique()*15),'ic':mean,'icir':icir,'hit_rate':hit,'turnover':turnover,'date_min':str(d.date.min().date()),'date_max':str(d.date.max().date())},indent=2))
os.makedirs('scripts',exist_ok=True)
pd.DataFrame(artifact).to_csv('scripts/miner_1_20261217_overnight_reversal_signal.csv',index=False)
for label, sub in [('2020_2022',d[(d.date<'2023-01-01')]),('2023_2024',d[(d.date>='2023-01-01')&(d.date<'2025-01-01')]),('2025_2026',d[d.date>='2025-01-01']),('recent120',d[d.date>=d.date.max()-pd.Timedelta(days=180)])]:
    a=[]
    for dt,g in sub.groupby('date'):
        if len(g)>=8 and g.signal.nunique()>1 and g.fwd1.nunique()>1:
            z=g.signal.corr(g.fwd1,method='spearman')
            if pd.notna(z):a.append(z)
    a=np.array(a); print(label,len(a),float(a.mean()) if len(a) else None,float(a.mean()/a.std(ddof=1)*np.sqrt(252)) if len(a)>1 and a.std(ddof=1)>0 else None)
for h in [5,10]:
    # calculate using close forward from source unavailable here; proxy compound consecutive fwd not valid, skip
    pass
