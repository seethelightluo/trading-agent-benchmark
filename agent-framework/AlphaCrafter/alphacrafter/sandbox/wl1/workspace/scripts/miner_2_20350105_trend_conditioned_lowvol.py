import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
p=pd.concat({s:load(s) for s in U},axis=1).sort_index().ffill(); r=p.pct_change()
# Candidate: low-volatility carry: inverse realized volatility, conditioned on positive 60d trend.
rv=r.rolling(30).std()*np.sqrt(30); trend=r.rolling(60).sum(); f=(1/rv.replace(0,np.nan))*(1+0.35*np.tanh(trend*5)); f=f.shift(1)
rows=[]
for i in range(len(p)-10):
 z=pd.concat([f.iloc[i],r.iloc[i+1:i+11].sum()],axis=1).dropna()
 if len(z)>=8: rows.append((p.index[i],len(z),z.iloc[:,0].corr(z.iloc[:,1])))
d=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); ic=d.ic.dropna()
print('candidate=trend_conditioned_lowvol_30d; dates',len(d),'avgN',d.n.mean(),'coverage',d.n.mean()/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2029-12-31'),('2030-01-01','2034-12-31'),('2034-01-01','2035-01-05')]:
 x=ic.loc[a:b];print(a,b,'dates',len(x),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1) if len(x)>1 else np.nan)
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'recentIC',ic.tail(260).mean(),'recentICIR',ic.tail(260).mean()/ic.tail(260).std(ddof=1))
f.to_csv('scripts/miner_2_20350105_trend_conditioned_lowvol_signal.csv')
