import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(sym):
    x=pd.read_csv(os.path.join(base,sym+'.csv'),parse_dates=['date']).set_index('date')
    return x['close'].replace(0,np.nan)
px=pd.concat({s:load(s) for s in U},axis=1).sort_index()
macro=pd.concat({s:pd.read_csv('../persistent/index_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in ['VIX','DXY']},axis=1).reindex(px.index).ffill()
ret=np.log(px).diff(); csmed=ret.rolling(20).sum().median(axis=1)
res=ret.rolling(20).sum().sub(csmed,axis=0)
vol=ret.rolling(40).std()*np.sqrt(252)
stress=(macro['VIX'].pct_change(10)>0)&(macro['DXY'].pct_change(10)>0)
# stress increases reversal strength; lag all observed inputs by one day
fac=(-res/vol).mul(np.where(stress,1.8,1.0),axis=0).shift(1)
rows=[]
for h in [10,20,40,60]:
    fwd=np.log(px.shift(-h)/px)
    ics=[]; dates=[]; ns=[]
    for d in px.index:
        a=fac.loc[d]; b=fwd.loc[d]; ok=a.notna()&b.notna()
        if ok.sum()>=8:
            ics.append(spearmanr(a[ok],b[ok]).statistic); dates.append(d); ns.append(ok.sum())
    z=pd.Series(ics,index=pd.DatetimeIndex(dates));
    print(h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(z.mean(),z.mean()/z.std(ddof=1), (z>0).mean(),len(z),np.mean(ns)))
    for lo,hi in [('2020','2023-12-31'),('2024','2026-12-31'),('2027','2029-12-31'),('2030','2032-12-31'),('2033','2034-03-02')]:
      q=z.loc[lo:hi]; print(' regime',lo,'%.6f %.6f n%d'%(q.mean(),q.mean()/q.std(ddof=1),len(q)))
# artifact at 10d horizon
out=pd.DataFrame(fac,index=px.index); out.index.name='date'; out.to_csv('scripts/miner_2_20340302_stress_residual_reversal_signal.csv')
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean().mean())
