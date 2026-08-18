import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
files=glob.glob('../persistent/stock_data/*.csv')
U={os.path.basename(f)[:-4]:pd.read_csv(f) for f in files}
for s,d in U.items():
    d['date']=pd.to_datetime(d['date']); d.sort_values('date',inplace=True); d['ret']=d['close'].pct_change(); U[s]=d.set_index('date')
dates=sorted(set.intersection(*[set(d.index) for d in U.values()]))
# factor: 20d return divided by downside deviation over trailing 20d, negative shocks penalize risk
rows=[]
for dt in dates:
    vals={}; fw={}
    for s,d in U.items():
        if dt not in d.index: continue
        x=d.loc[:dt].tail(25)
        if len(x)<21: continue
        r=x['close'].iloc[-1]/x['close'].iloc[-21]-1
        neg=x['ret'].iloc[-20:]; down=np.sqrt(np.mean(np.minimum(neg,0)**2))
        if np.isfinite(r) and np.isfinite(down): vals[s]=r/(down+0.005)
        fut=d.loc[d.index>dt,'close'].head(10)
        if len(fut)>=10: fw[s]=fut.iloc[-1]/d.loc[dt,'close']-1
    common=set(vals)&set(fw)
    if len(common)>=8:
        a=np.array([vals[s] for s in common]); b=np.array([fw[s] for s in common])
        ic=spearmanr(a,b).statistic
        rows.append((dt,ic,len(common),len(vals)))
r=pd.DataFrame(rows,columns=['date','ic','n','nv']).set_index('date')
for label,z in [('all',r),('recent500',r.tail(500)),('recent250',r.tail(250))]:
    ic=z.ic.mean(); sd=z.ic.std(ddof=1); print(label,'dates',len(z),'avgN',z.n.mean(),'IC',round(ic,6),'ICIR',round(ic/sd,6),'hit',round((z.ic>0).mean(),4),'coverage',round(z.n.mean()/15,4))
print('turnover proxy unavailable; latest',r.tail(1).index[0], 'range',r.index[0],r.index[-1])
