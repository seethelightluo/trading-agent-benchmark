import os, numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
px={}
for s in U:
    f=os.path.join(root,s+'.csv')
    if os.path.exists(f):
        d=pd.read_csv(f); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); rets=prices.pct_change()
# Upside/downside efficiency: reward consistency relative to harmful days, lagged one day
up=rets.clip(lower=0).rolling(30,min_periods=20).sum()
dn=(-rets.clip(upper=0)).rolling(30,min_periods=20).sum()
factor=(up-dn)/(up+dn+1e-12)
# forward 10 trading-day return, factor observed t and return t+1..t+10
fwd=prices.shift(-10)/prices-1
rows=[]
for dt in factor.index:
    x=factor.loc[dt]; y=fwd.loc[dt]
    z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avg_n',r.n.mean(),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean())
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=r.loc[a:b]
 print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan,(q.ic>0).mean())
for h in [5,10,20]:
 fy=prices.shift(-h)/prices-1; rr=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('h',h,'n',len(rr),'ic',np.nanmean(rr),'icir',np.nanmean(rr)/np.nanstd(rr,ddof=1))
# turnover rank changes
rank=factor.rank(axis=1,pct=True); turnover=(rank-rank.shift(1)).abs().mean(axis=1).dropna().mean()
print('coverage',factor.notna().mean().mean(),'rank_turnover',turnover)
out=factor.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_1_20320916_updown_efficiency_signal.csv',index=False)
