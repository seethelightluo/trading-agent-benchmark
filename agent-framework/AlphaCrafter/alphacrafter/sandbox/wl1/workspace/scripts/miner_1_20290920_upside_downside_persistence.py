import os, json
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Upside/downside persistence: lagged 40d positive-return mass relative to negative mass.
rows=[]
for s in watch:
    df=get_stock_daily_data(s, days=5000)
    if df is None or len(df)<60: continue
    df=df.sort_values('date').copy(); r=pd.to_numeric(df['pct_change'],errors='coerce')/100.0
    # exclude current observation: shift returns before rolling
    pos=r.clip(lower=0).shift(1).rolling(40,min_periods=30).sum()
    neg=(-r.clip(upper=0)).shift(1).rolling(40,min_periods=30).sum()
    # signed asymmetry, stabilized; higher means returns generated with upside persistence
    f=(pos-neg)/(pos+neg+1e-10)
    out=pd.DataFrame({'date':pd.to_datetime(df.date),'symbol':s,'factor':f.values,'close':df.close.values})
    out['fwd10']=df.close.shift(-10).values/df.close.values-1
    rows.append(out)
x=pd.concat(rows,ignore_index=True).dropna(subset=['factor','fwd10'])
# ensure 10d factor same-date and future return; persist artifact
os.makedirs('scripts',exist_ok=True)
artifact='scripts/miner_1_20290920_upside_downside_persistence_signal.csv'
x[['date','symbol','factor']].to_csv(artifact,index=False)
ics=[]; counts=[]
for d,g in x.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:
        ics.append(g.factor.corr(g.fwd10,method='spearman')); counts.append(len(g))
a=pd.Series(ics).dropna()
print(json.dumps({'factor':'upside_downside_persistence_40','artifact':artifact,'valid_dates':int(len(a)),'avg_instruments':float(np.mean(counts)),'coverage':float(x.factor.notna().mean()),'IC':float(a.mean()),'ICIR':float(a.mean()/a.std(ddof=1)),'hit_ratio':float((a>0).mean()),'turnover':float(x.sort_values(['symbol','date']).groupby('symbol').factor.apply(lambda z:(z.diff().abs()>0.05).mean()).mean())},indent=2))
# regimes
x['year']=pd.to_datetime(x.date).dt.year
for label,mask in [('2020-25',(x.year<=2025)),('2026+',(x.year>=2026)),('2028+',(x.year>=2028)),('2029YTD',(x.year==2029))]:
    z=x[mask]; q=[]
    for d,g in z.groupby('date'):
        if len(g)>=8 and g.factor.nunique()>1 and g.fwd10.nunique()>1:q.append(g.factor.corr(g.fwd10,method='spearman'))
    q=pd.Series(q).dropna(); print(label,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None)
