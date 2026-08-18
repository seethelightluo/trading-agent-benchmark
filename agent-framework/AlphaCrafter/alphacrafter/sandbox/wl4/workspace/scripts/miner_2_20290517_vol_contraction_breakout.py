import os, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
frames={}
for s in U:
    p=os.path.join(base,s+'.csv')
    if os.path.exists(p):
        d=pd.read_csv(p)
        d['date']=pd.to_datetime(d['date'])
        d=d.sort_values('date').drop_duplicates('date').set_index('date')
        frames[s]=d
# factor: directional 10d move, volatility-normalized, rewarded when 20d vol is compressed vs 60d baseline
rows=[]
for s,d in frames.items():
    c=pd.to_numeric(d['close'],errors='coerce')
    r=c.pct_change()
    v20=r.rolling(20,min_periods=15).std()
    v60=r.rolling(60,min_periods=40).std()
    f=(c.pct_change(10)/(v20*np.sqrt(10))).replace([np.inf,-np.inf],np.nan) * (v60/v20).clip(0.5,2.0)
    # cap to avoid outlier domination
    f=f.clip(-8,8)
    z=pd.DataFrame({'factor':f,'fwd1':c.pct_change(-1),'fwd5':c.pct_change(-5),'fwd10':c.pct_change(-10),'fwd20':c.pct_change(-20),'asset':s})
    rows.append(z.reset_index())
x=pd.concat(rows,ignore_index=True)
cut=pd.Timestamp('2029-05-16'); x=x[x.date<=cut]

def calc(col):
    vals=[]; ns=[]
    for dt,g in x.groupby('date'):
        g=g[['factor',col]].dropna()
        if len(g)>=8:
            vals.append(spearmanr(g.factor,g[col]).statistic); ns.append(len(g))
    a=np.asarray(vals); return len(a),float(np.nanmean(a)),float(np.nanmean(a)/np.nanstd(a,ddof=1)),float(np.mean(a>0)),float(np.mean(ns)),float(np.mean(ns)/15)
print('cutoff',cut.date(),'rows',len(x),'assets',x.asset.nunique())
for h in ['fwd1','fwd5','fwd10','fwd20']:
    print(h, 'dates IC ICIR hit avgN coverage',calc(h))
# turnover proxy: average fraction ranks changed among consecutive common dates
rank=x.dropna(subset=['factor']).pivot(index='date',columns='asset',values='factor').rank(axis=1,pct=True)
print('rank_turnover',float(rank.diff().abs().mean(axis=1).dropna().mean()))
# recent 250 daily observations
vals=[]
for dt,g in x.groupby('date'):
 g=g[['factor','fwd1']].dropna()
 if len(g)>=8: vals.append((dt,spearmanr(g.factor,g.fwd1).statistic))
print('recent250_daily',len(vals),float(np.mean([v for _,v in vals[-250:]])),float(np.mean([v for _,v in vals[-250:]])/np.std([v for _,v in vals[-250:]],ddof=1)))
