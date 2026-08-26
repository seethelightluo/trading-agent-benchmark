import os, sys
import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cutoff=pd.Timestamp('2032-09-05')
# positive-return breadth: at each date, use returns ending at t, factor is available next day
px={}
for s in U:
    d=get_stock_daily_data(s, days=5000)
    if d is not None and len(d):
        x=d.copy(); x['date']=pd.to_datetime(x['date']); x=x[x.date<=cutoff].set_index('date')['close'].astype(float)
        px[s]=x
P=pd.DataFrame(px).sort_index().ffill()
r20=P/P.shift(20)-1
vol20=P.pct_change().rolling(60).std()
# lag regime and lag signal at t; forward return begins t+1
breadth=(r20>0).mean(axis=1)
# blended threshold avoids same-day leakage by factor itself only t information
reg=(2*breadth-1)
F=r20.mul(reg, axis=0).div(vol20*np.sqrt(20))
F=F.replace([np.inf,-np.inf],np.nan)
rets=P.shift(-1)/P-1

def calc(h):
    fr=P.shift(-h)/P-1
    vals=[]; ns=[]; turnover=[]
    dates=F.index
    for dt in dates:
        a=F.loc[dt]; b=fr.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
        if len(z)>=8:
            vals.append(z.iloc[:,0].corr(z.iloc[:,1])); ns.append(len(z))
    ic=pd.Series(vals)
    # turnover in rank signal on consecutive valid dates
    ranks=F.rank(axis=1,pct=True); dif=ranks.diff().abs().mean(axis=1)
    return len(ic),float(ic.mean()),float(ic.mean()/ic.std(ddof=1)) if ic.std(ddof=1)>0 else np.nan,float((F.notna().sum().sum())/(F.shape[0]*len(U))),float(dif.mean()),float(np.mean(ic>0)),ns
print('cutoff',cutoff.date(),'dates',len(P),'assets',len(px),'avgN',round(F.notna().sum(axis=1).mean(),2))
for h in [1,5,10,20]: q=calc(h); print('H',h,'n',q[0],'IC',round(q[1],6),'ICIR',round(q[2],6),'coverage',round(q[3],4),'turnover',round(q[4],4),'hit',round(q[5],4))
# thirds for H10 and H20
for h in [10,20]:
    fr=P.shift(-h)/P-1; ics=[]
    for dt in F.index:
        z=pd.concat([F.loc[dt],fr.loc[dt]],axis=1).dropna()
        if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1])))
    q=len(ics)//3
    print('thirds',h,[round(np.mean([v for _,v in ics[i*q:(i+1)*q]]),5) for i in range(3)])
# artifact full factor
out=F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna()
out.to_csv('scripts/miner_3_20320906_positive_breadth20_signal.csv',index=False)
print('artifact',len(out))
