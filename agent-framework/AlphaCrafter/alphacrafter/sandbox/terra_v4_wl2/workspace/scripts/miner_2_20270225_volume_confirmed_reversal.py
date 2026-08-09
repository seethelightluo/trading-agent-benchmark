import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index() for a in A}
# Volume-confirmed reversal: short-term reversal is stronger when the recent move occurred on unusually high volume.
sig={}; close={}
for a in A:
    c=d[a].close; v=d[a].volume.replace(0,np.nan)
    move=c.pct_change(3)
    vz=np.log(v).sub(np.log(v).rolling(60,min_periods=30).mean()).clip(-3,3)
    sig[a]=-move*(1+vz.clip(lower=0))
    close[a]=c

def run(h):
    out=[]; ns=[]; dates=sorted(set().union(*[close[a].index for a in A]))
    for dt in dates:
        x=[]; y=[]
        for a in A:
            if dt in sig[a].index:
                x.append(sig[a].loc[dt]); y.append(close[a].shift(-h).get(dt)/close[a].get(dt)-1)
        z=pd.DataFrame({'x':x,'y':y}).dropna()
        if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1:
            q=spearmanr(z.x,z.y).statistic
            if np.isfinite(q): out.append(q); ns.append(len(z))
    q=np.asarray(out); print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(len(q)/len(dates),4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
    for name,mask in [('2020-22',lambda s:s<'2023'),('2023-24',lambda s:'2023'<=s<'2025'),('2025-26',lambda s:'2025'<=s<'2027'),('2026-07+',lambda s:s>='2026-07')]:
        # use aligned dates again for regime stats
        vals=[]
        for dt in dates:
            if not mask(str(dt)[:10]): continue
            xx=[]; yy=[]
            for a in A:
                if dt in sig[a].index: xx.append(sig[a].loc[dt]); yy.append(close[a].shift(-h).get(dt)/close[a].get(dt)-1)
            z=pd.DataFrame({'x':xx,'y':yy}).dropna()
            if len(z)>=8 and z.x.nunique()>1 and z.y.nunique()>1: vals.append(spearmanr(z.x,z.y).statistic)
        if vals: print(name,len(vals),round(np.mean(vals),6),round(np.mean(vals)/np.std(vals,ddof=1),6))
for h in [1,5,10]: run(h)
print('max_abs_library_correlation',None)
