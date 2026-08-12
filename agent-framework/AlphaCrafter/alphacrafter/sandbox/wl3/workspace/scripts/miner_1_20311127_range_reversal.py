import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=None
    for fn in (get_index_daily_data,get_stock_daily_data):
        try:
            z=fn(s,days=5000)
            if z is not None and len(z)>0: x=z; break
        except Exception: pass
    if x is not None:
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index()
        D[s]=x
print('assets',len(D),{k:len(v) for k,v in D.items()})
# range-normalized short reversal: negative 5d log return scaled by 20d true-range proxy
series={}
for s,x in D.items():
    c=pd.to_numeric(x.close,errors='coerce')
    hi=pd.to_numeric(x.high,errors='coerce'); lo=pd.to_numeric(x.low,errors='coerce')
    tr=(hi-lo)/c
    r5=np.log(c/c.shift(5)); atr=tr.rolling(20,min_periods=15).mean()
    # activation by range expansion, clipped for robustness
    f=(-r5/atr).clip(-10,10)
    series[s]=pd.DataFrame({'f':f,'ret':np.log(c.shift(-10)/c)})
dates=sorted(set().union(*[set(z.index) for z in series.values()]))
ics=[]; breadth=[]; turnover=[]; prev=None
for d in dates:
    a=[]
    for s,z in series.items():
        if d in z.index and np.isfinite(z.loc[d,'f']) and np.isfinite(z.loc[d,'ret']): a.append((s,z.loc[d,'f'],z.loc[d,'ret']))
    if len(a)>=8:
        q=pd.DataFrame(a,columns=['s','f','r']); ic=q.f.corr(q.r,method='spearman')
        if np.isfinite(ic):
            ics.append((d,ic,len(a))); breadth.append(len(a)/15)
            ranks=q.set_index('s').f.rank(pct=True); cur=ranks
            if prev is not None: turnover.append(np.mean([abs(cur.get(s,np.nan)-prev.get(s,np.nan)) for s in set(cur.index)|set(prev.index) if np.isfinite(cur.get(s,np.nan)) and np.isfinite(prev.get(s,np.nan))]))
            prev=cur
v=np.array([x[1] for x in ics]); by=pd.DataFrame(ics,columns=['date','ic','n'])
print('dates',len(v),'avg_n',np.mean(by.n),'coverage',np.mean(breadth),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',np.mean(v>0),'turnover',np.mean(turnover))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2031')]:
 q=by[(by.date.astype(str)>=a)&(by.date.astype(str)<=b+'-12-31')].ic
 print(a,b,'n',len(q),'IC',q.mean() if len(q) else None,'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else None)
# artifact
out=[]
for d,ic,n in ics:
 for s,z in series.items():
  if d in z.index and np.isfinite(z.loc[d,'f']): out.append({'date':d,'symbol':s,'signal':z.loc[d,'f']})
pd.DataFrame(out).to_csv('scripts/miner_1_20311127_range_reversal_signal.csv',index=False)
