import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_index_daily_data(s,days=3200)
    except Exception: x=get_stock_daily_data(s,days=3200)
    if x is not None and len(x)>100: D[s]=x.sort_values('date').set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# lagged 60-session volatility-adjusted contrarian score; higher means weaker prior risk-adjusted return
v=r.rolling(60,min_periods=40).std()*np.sqrt(60)
f=-(p.pct_change(60)/v)
print('data_dates',len(p),'assets',len(D),'range',p.index.min(),p.index.max())
for h in [1,5,10,20]:
    ics=[]; ns=[]; recent=[]; early=[]; mid=[]
    for i in range(1,len(p)-h):
        q=pd.concat([f.iloc[i-1].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('r')],axis=1).dropna()
        if len(q)>=8:
            c=q.f.corr(q.r)
            if np.isfinite(c):
                ics.append(c); ns.append(len(q))
                if i>=len(p)-250: recent.append(c)
                elif i< len(p)//2: early.append(c)
                else: mid.append(c)
    a=np.asarray(ics)
    def stat(x):
        x=np.asarray(x); return (float(x.mean()),float(x.mean()/x.std(ddof=1))) if len(x)>1 else (np.nan,np.nan)
    print({'h':h,'dates':len(a),'avg_n':round(np.mean(ns),2),'IC':round(stat(a)[0],6),'ICIR':round(stat(a)[1],6),'hit':round(np.mean(a>0),4),'coverage':round(np.mean(ns)/15,4),'early':tuple(round(z,6) for z in stat(early)),'mid':tuple(round(z,6) for z in stat(mid)),'recent250':tuple(round(z,6) for z in stat(recent))})
# rank turnover proxy
z=f.rank(axis=1,pct=True); print('turnover',round(float((z.diff().abs().stack().mean())),6))
