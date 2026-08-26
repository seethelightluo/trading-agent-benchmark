import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr

CUT=pd.Timestamp('2034-10-02')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
    f=f'../persistent/stock_data/{s}.csv'
    d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()
    px[s]=d['close'].loc[:CUT]
prices=pd.DataFrame(px).sort_index()
ret=prices.pct_change()
# Candidate: range/volatility-normalized short-term reversal. Scale recent 3d loss
# by trailing 20d return volatility, with winsorization only for stable ranks.
signal=-(prices.pct_change(3))/ret.rolling(20,min_periods=15).std()
rows=[]; artifact=[]
for dt in signal.index:
    fwd=prices.shift(-5).loc[dt]/prices.loc[dt]-1
    x=signal.loc[dt]
    z=pd.concat([x.rename('x'),fwd.rename('y')],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.x,z.y).statistic
        rows.append((dt,len(z),ic))
        for s in z.index: artifact.append((dt,s,float(x[s]),float(fwd[s])))
r=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('candidate=volatility_scaled_3d_reversal; cutoff',CUT.date())
print('dates',len(r),'avg_n',round(r.n.mean(),2),'coverage',round(r.n.mean()/15,4))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit', (r.ic>0).mean(),'median',r.ic.median())
for a,b in [('2020','2022'),('2023','2026'),('2027','2030'),('2031','2034')]:
 q=r.loc[a:b].ic
 print(a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,'hit',(q>0).mean())
# rough rank turnover
rank=signal.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna()
print('rank_turnover_proxy',turn.mean())
os.makedirs('scripts',exist_ok=True)
pd.DataFrame(artifact,columns=['date','symbol','signal','forward_5d_return']).to_csv('scripts/miner_1_20341002_volscaled_reversal_signal.csv',index=False)
print('artifact rows',len(artifact))
# decay
for h in [1,5,10,20]:
 vals=[]
 for dt in signal.index:
  f=prices.shift(-h).loc[dt]/prices.loc[dt]-1
  z=pd.concat([signal.loc[dt].rename('x'),f.rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic)
 print('horizon',h,'dates',len(vals),'IC',np.mean(vals),'ICIR',np.mean(vals)/np.std(vals,ddof=1))
