import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    try: x=get_stock_daily_data(s, days=3600)
    except Exception as e: print('missing',s,str(e)); continue
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); x=x.set_index('date').sort_index(); D[s]=x['close'].astype(float)
P=pd.DataFrame(D).sort_index().ffill(); r=P.pct_change()
ret=P/P.shift(60)-1; vol=r.rolling(60).std()*np.sqrt(252); eff=ret.abs()/(r.abs().rolling(60).sum()+1e-12); f=(ret/(vol+1e-12))*eff
rows=[]
for i in range(1,len(P)-10):
    vals=f.iloc[i-1]; fw=P.iloc[i+10]/P.iloc[i]-1; z=pd.concat([vals.rename('f'),fw.rename('y')],axis=1).dropna()
    if len(z)>=8: rows.append((P.index[i],z.f.corr(z.y,method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for n in [120,260,520,780,1200,len(a)]:
 b=a.tail(n); mu=b.ic.mean(); sd=b.ic.std(ddof=1); print(n,'dates',len(b),'avg_n',round(b.n.mean(),2),'IC',round(mu,6),'ICIR',round(mu/sd,6) if sd else 0,'hit',round((b.ic>0).mean(),4))
valid=f.notna().sum(axis=1)/15; rank=f.rank(axis=1,pct=True); print('TOTAL_DATES',len(a),'period',a.index.min(),a.index.max(),'coverage',round(valid.mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4))
for start,end in [('2020','2023'),('2023','2027'),('2027','2031'),('2031','2034')]:
 b=a.loc[start:end]; print('REGIME',start,end,len(b),round(b.ic.mean(),6),round(b.ic.mean()/b.ic.std(ddof=1),6) if len(b)>2 else None)
f.loc[a.index].to_csv('scripts/artifacts/miner_3_20340608_trend_quality_signal.csv'); a.to_csv('scripts/artifacts/miner_3_20340608_trend_quality_ic.csv')
