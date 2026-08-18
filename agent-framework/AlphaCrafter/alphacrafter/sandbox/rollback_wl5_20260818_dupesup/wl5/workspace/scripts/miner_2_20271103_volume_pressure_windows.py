import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: D[s]=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
    except Exception as e: print('missing',s,e)

def signal(x,window):
    rng=(x.high-x.low).replace(0,np.nan)
    clv=(2*x.close-x.high-x.low)/rng
    vr=(x.volume/x.volume.rolling(20,min_periods=10).median()).clip(0,3)
    return -(clv*vr).rolling(window,min_periods=window).mean()

def evaluate(window):
    rows=[]
    for s,x in D.items():
        f=signal(x,window)
        # only information available at t, predict t+1
        for i in range(len(x)-1):
            if pd.notna(f.iloc[i]) and pd.notna(x.close.iloc[i]) and pd.notna(x.close.iloc[i+1]):
                rows.append((x.index[i],s,float(f.iloc[i]),float(x.close.iloc[i+1]/x.close.iloc[i]-1)))
    a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).query("date >= '2020-01-01' and date <= '2027-10-29'")
    ics=[]; ns=[]
    for dt,g in a.groupby('date'):
        if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
            ics.append(spearmanr(g.factor,g.fwd).statistic); ns.append(len(g))
    z=np.asarray(ics,float)
    # date-level signal turnover: rank changes for overlapping symbols
    piv=a.pivot(index='date',columns='symbol',values='factor'); ranks=piv.rank(axis=1,pct=True)
    turn=ranks.diff().abs().mean(axis=1).mean()
    print({'window':window,'dates':len(z),'mean_n':round(float(np.mean(ns)),2),'coverage':round(a.symbol.nunique()/15,3),'IC':round(float(np.nanmean(z)),6),'ICIR':round(float(np.nanmean(z)/np.nanstd(z,ddof=1)),6),'hit':round(float(np.mean(z>0)),4),'turnover_rank':round(float(turn),4)})
    for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-10-29')]:
        q=a[(a.date>=lo)&(a.date<=hi)]; zz=[]
        for _,g in q.groupby('date'):
            if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: zz.append(spearmanr(g.factor,g.fwd).statistic)
        print(' regime',lo,round(float(np.nanmean(zz)),6),len(zz))
for w in [3,5,10]: evaluate(w)
# save artifact for the strongest candidate; persistence handled only after gate
best=signal(D[next(iter(D))],3)
print('symbols',len(D))
