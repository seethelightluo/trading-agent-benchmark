import os, numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    p=os.path.join(base,s+'.csv')
    d=pd.read_csv(p)
    d['date']=pd.to_datetime(d['date'])
    d=d.sort_values('date').set_index('date')
    px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# Candidate: recovery from trailing 60-day high, with a 5-day confirmation filter.
# Higher value means closer to high and improving over the last 5 sessions.
rollmax=P.rolling(60,min_periods=40).max()
raw=P/rollmax-1
f=raw + 0.5*(P/P.shift(5)-1)
f=f.replace([np.inf,-np.inf],np.nan)
fwd=P.shift(-10)/P-1
rows=[]
for dt in f.index:
    a=f.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=drawdown_recovery_60d_confirm5')
print('dates',len(r),'mean_n',r.n.mean(),'coverage',r.n.sum()/(len(r)*15))
print('IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(ddof=1), (r.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2027'),('2027','2028')]:
 x=r.loc[a:b]
 if len(x): print(a+'-'+b,'dates',len(x),'IC %.8f ICIR %.8f'%(x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1)))
# rank turnover: average fraction changing top/bottom ordering proxy
ranks=f.rank(axis=1,pct=True)
turn=(ranks.diff().abs().mean(axis=1)).dropna().mean()
print('mean_rank_change',turn)
for h in [1,5,10,20]:
 rr=[]; ff=P/rollmax-1+0.5*(P/P.shift(5)-1)
 fh=P.shift(-h)/P-1
 for dt in ff.index:
  z=pd.concat([ff.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('horizon',h,'IC',np.nanmean(rr),'dates',len(rr))
# artifact
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20280114_drawdown_recovery_60d_signal.csv',index=False)
