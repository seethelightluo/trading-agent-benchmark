import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for a in assets:
 p=os.path.join(base,a+'.csv')
 if os.path.exists(p): px[a]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
r=np.log(P).diff(); dr=np.log(dxy).diff(); mom=np.log(P/P.shift(20)); vol=r.rolling(20).std()*np.sqrt(20)
dxytrend=dr.rolling(20).sum(); cond=(1-np.tanh(dxytrend/0.03)).shift(1)
f=mom.div(vol).mul(cond,axis=0)
rows=[]
for i in range(len(P)-10):
 x=f.iloc[i]; y=(P.iloc[i+10]/P.iloc[i]-1); ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((P.index[i],spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic.dropna(); mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd*np.sqrt(252)
print('dates',len(z),'assets',len(px),'avgN',z.n.mean(),'coverage',z.n.mean()/15); print('IC',mean,'ICIR',icir,'hit',(ic>0).mean())
for w in [120,252,756]:
 q=ic.tail(w); print('recent',w,q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252))
for a,b in [('2020','2025'),('2026','2030'),('2031','2032')]:
 q=ic[(ic.index>=a)&(ic.index<=b+'-12-31')]; print('regime',a,b,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>2 else np.nan)
ranks=f.rank(axis=1,pct=True); print('turnover',((ranks-ranks.shift(1)).abs().mean(axis=1)).mean())
f.stack().rename('signal').to_csv('factors/miner_1_20321111_dxy_conditioned_momentum20_signal.csv'); z.to_csv('factors/miner_1_20321111_dxy_conditioned_momentum20_ic.csv')
