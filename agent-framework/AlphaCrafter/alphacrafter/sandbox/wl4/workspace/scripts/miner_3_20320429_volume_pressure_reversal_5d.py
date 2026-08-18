import os, json
import numpy as np
import pandas as pd

CUTOFF = pd.Timestamp('2032-04-29')
H = 10
files = [f for f in os.listdir('../persistent/stock_data') if f.endswith('.csv')]
prices = {}
for f in files:
    s=f[:-4]; d=pd.read_csv('../persistent/stock_data/'+f, parse_dates=['date']).sort_values('date')
    d=d[d.date<=CUTOFF].set_index('date')
    # signed intraday pressure, volume relative to trailing 20d median
    intraday=np.log(d['close']/d['open'].replace(0,np.nan))
    vratio=d['volume']/d['volume'].rolling(20,min_periods=10).median().replace(0,np.nan)
    pressure=(intraday*vratio).rolling(5,min_periods=5).sum()
    # negative pressure is contrarian; standardize by recent pressure volatility
    scale=intraday.rolling(20,min_periods=15).std()*np.sqrt(5)
    prices[s]=pd.DataFrame({'fac':-pressure/scale.replace(0,np.nan),'close':d.close})

all_dates=sorted(set().union(*[set(x.index) for x in prices.values()]))
ics=[]; ns=[]; vals=[]; turnover=[]
prev={}
for dt in all_dates:
    rows=[]
    for s,x in prices.items():
        if dt not in x.index: continue
        i=x.index.get_loc(dt)
        if i+H>=len(x): continue
        f=x.iloc[i].fac; c=x.iloc[i].close; cnext=x.iloc[i+H].close
        if np.isfinite(f) and c>0 and cnext>0: rows.append((s,float(f),float(cnext/c-1)))
    if len(rows)>=8:
        a=pd.DataFrame(rows,columns=['s','f','r']).dropna()
        if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:
            ic=a.f.corr(a.r); ics.append(ic); ns.append(len(a)); vals.append((dt,ic,len(a)))
            cur={r[0]:r[1] for r in rows}; common=set(cur)&set(prev)
            if common: turnover.append(np.mean([abs(cur[s]-prev[s]) for s in common]))
            prev=cur
arr=np.array(ics); mean=float(np.nanmean(arr)); sd=float(np.nanstd(arr,ddof=1)); icir=mean/sd*np.sqrt(252) if sd else np.nan
print(json.dumps({'candidate':'volume_pressure_reversal_5d','cutoff':str(CUTOFF.date()),'horizon':H,'dates':len(arr),'avg_n':float(np.mean(ns)),'min_n':int(np.min(ns)),'coverage':float(sum(ns)/(len(arr)*15)),'ic':mean,'icir':icir,'hit_ratio':float(np.mean(arr>0)),'turnover_proxy':float(np.mean(turnover)) if turnover else None,'recent_365_ic':float(np.mean([x[1] for x in vals if x[0]>=CUTOFF-pd.Timedelta(days=365)])),'recent_365_dates':sum(x[0]>=CUTOFF-pd.Timedelta(days=365) for x in vals)},indent=2))
for h in [5,20]:
    z=[]
    for dt in all_dates:
      rows=[]
      for s,x in prices.items():
       if dt in x.index:
        i=x.index.get_loc(dt)
        if i+h<len(x) and np.isfinite(x.iloc[i].fac): rows.append((x.iloc[i].fac,x.iloc[i+h].close/x.iloc[i].close-1))
      if len(rows)>=8: z.append(pd.DataFrame(rows,columns=['f','r']).corr().iloc[0,1])
    print('decay',h,'dates',len(z),'ic',float(np.nanmean(z)) if z else None,'icir',float(np.nanmean(z)/np.nanstd(z,ddof=1)*np.sqrt(252)) if len(z)>1 else None)
