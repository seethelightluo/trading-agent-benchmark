import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 try:x=get_stock_daily_data(s,days=3600)
 except:continue
 if x is not None and len(x): x=x.assign(date=pd.to_datetime(x.date)).set_index('date').sort_index();D[s]=x.close.astype(float)
P=pd.DataFrame(D).sort_index().ffill(); r=P.pct_change(); f=-r.rolling(20).std(); rows=[]
for i in range(1,len(P)-10):
 z=pd.concat([f.iloc[i-1].rename('f'),(P.iloc[i+10]/P.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8:rows.append((P.index[i],z.f.corr(z.y,method='spearman'),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for n in [120,260,520,780,1200,len(a)]:
 b=a.tail(n);print(n,len(b),round(b.n.mean(),2),round(b.ic.mean(),6),round(b.ic.mean()/b.ic.std(ddof=1),6),round((b.ic>0).mean(),4))
print('TOTAL',len(a),a.index.min(),a.index.max(),'coverage',round(f.notna().sum(axis=1).mean()/15,4),'turn',round(f.rank(pct=True).diff().abs().mean(axis=1).mean(),4))
for st,en in [('2020','2023'),('2023','2027'),('2027','2031'),('2031','2034')]:
 b=a.loc[st:en];print('REGIME',st,en,len(b),round(b.ic.mean(),6),round(b.ic.mean()/b.ic.std(ddof=1),6))
f.loc[a.index].to_csv('scripts/artifacts/miner_3_20340608_lowvol_signal.csv');a.to_csv('scripts/artifacts/miner_3_20340608_lowvol_ic.csv')
