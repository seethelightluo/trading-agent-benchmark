import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2026-07-15'); D={}
for s in U:
    try:
        x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
        D[s]=x.loc[x.index<=cutoff]
    except Exception as e: print('missing',s,e)
# One interpretable idea: overnight gap reversal. Signal is negative prior-session open-to-close gap
# (today open / yesterday close - 1), point-in-time at today's close; forward return is next close / today's close.
# This is evaluated cross-sectionally and avoids using tomorrow data.
rows=[]
for s,x in D.items():
    gap=x.open/x.close.shift(1)-1
    f=-gap
    y=x.close.shift(-1)/x.close-1
    for dt in x.index:
        if pd.notna(f.loc[dt]) and pd.notna(y.loc[dt]): rows.append((dt,s,float(f.loc[dt]),float(y.loc[dt])))
a=pd.DataFrame(rows,columns=['date','symbol','factor','forward'])
ics=[]; ns=[]
for dt,g in a.groupby('date'):
    if len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1:
        ics.append(spearmanr(g.factor,g.forward).statistic); ns.append(len(g))
z=np.asarray(ics)
print('dates',len(z),'avg_names',round(np.mean(ns),3),'symbols',a.symbol.nunique(),'coverage',a.symbol.nunique()/15)
print('daily IC %.8f ICIR %.8f hit %.4f std %.8f'%(z.mean(),z.mean()/z.std(ddof=1),np.mean(z>0),z.std(ddof=1)))
for h in [5,10]:
    rr=[]
    for s,x in D.items():
        gap=- (x.open/x.close.shift(1)-1); y=x.close.shift(-h)/x.close-1
        for dt in x.index:
            if pd.notna(gap.loc[dt]) and pd.notna(y.loc[dt]): rr.append((dt,s,float(gap.loc[dt]),float(y.loc[dt])))
    q=pd.DataFrame(rr,columns=['date','s','f','y']); v=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1:v.append(spearmanr(g.f,g.y).statistic)
    v=np.array(v); print('%dd IC %.8f ICIR %.8f dates %d'%(h,v.mean(),v.mean()/v.std(ddof=1),len(v)))
# rank turnover and broad recent/regime stability
r=a.assign(rank=a.groupby('date').factor.rank(pct=True)).pivot(index='date',columns='symbol',values='rank').sort_index()
print('rank_turnover',r.diff().abs().mean().mean())
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]:
    q=z[(r.index>=lo)&(r.index<=hi)] if False else None
    # recompute subset from date-indexed IC list
    sub=[]
    for dt,g in a.groupby('date'):
        if lo<=str(dt.date())<=hi and len(g)>=8 and g.factor.nunique()>1 and g.forward.nunique()>1: sub.append(spearmanr(g.factor,g.forward).statistic)
    sub=np.asarray(sub); print(label,'dates',len(sub),'IC',sub.mean(),'ICIR',sub.mean()/sub.std(ddof=1) if len(sub)>1 else np.nan)
