import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def rd(s,macro=False):
 p='../persistent/index_data/' if macro else '../persistent/stock_data/'
 x=pd.read_csv(p+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); return x.close
px=pd.concat({s:rd(s) for s in U},axis=1).sort_index(); r=px.pct_change();
# DXY-conditioned reversal: use only yesterday's DXY shock to scale 3d cross-sectional reversal
m=rd('DXY',True).reindex(px.index).ffill(); shock=m.pct_change().rolling(3).sum().shift(1)
# stronger reversal after dollar shocks, with smooth positive scale
scale=(1+10*shock.abs()).clip(upper=3)
f=-r.rolling(3).sum().mul(scale,axis=0); f=f.sub(f.median(axis=1),axis=0)
rows=[]
for d in f.index:
 z=pd.concat([f.loc[d],(px.shift(-1)/px-1).loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']); print('dates',len(x),'avgN',x.n.mean(),'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean(),'coverage',x.n.mean()/15)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027-02-25')]:
 q=x.set_index('date').loc[lo:hi].ic; print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1))
out=f.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('../persistent/factor_signals_miner_3_20270225_dxy_shock_reversal.csv',index=False)
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
