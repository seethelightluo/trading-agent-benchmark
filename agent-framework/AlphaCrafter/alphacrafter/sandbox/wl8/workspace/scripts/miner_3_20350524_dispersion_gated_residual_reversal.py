import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-05-24')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].astype(float)
px=pd.concat({a:load(a) for a in assets},axis=1).sort_index().loc[:END].ffill()
r=px.pct_change(); bench=r.mean(axis=1); rr=r.sub(bench,axis=0)
# Short-horizon residual reversal, scaled by idiosyncratic volatility.
res5=(1+rr).rolling(5).apply(np.prod,raw=True)-1
vol20=rr.rolling(20).std()
base=-res5/vol20.clip(lower=vol20.mean(axis=1)*0.25,axis=0)
# Activate the reversal only when cross-asset dispersion is elevated; lag all inputs.
disp=rr.std(axis=1).rolling(20).rank(pct=True)
gate=(0.5+disp).clip(0.5,1.5)
sig=base.mul(gate,axis=0).shift(1)
rows=[]
for dt in px.index:
 x=sig.loc[dt]; y=px.shift(-10).loc[dt]/px.loc[dt]-1; ok=x.notna()&y.notna()
 if ok.sum()>=8:
  z=spearmanr(x[ok],y[ok]).statistic
  if np.isfinite(z): rows.append((dt,z,int(ok.sum())))
ic=np.array([q[1] for q in rows]); mean=ic.mean(); sd=ic.std(ddof=1)
turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).reindex([q[0] for q in rows]).mean()
print('period',rows[0][0].date(),rows[-1][0].date(),'dates',len(rows),'avgN',np.mean([q[2] for q in rows]))
print('IC10',mean,'ICIR_daily',mean/sd*np.sqrt(252),'hit',np.mean(ic>0),'turnover',turn,'coverage',np.mean([q[2]/15 for q in rows]))
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; out=[]
 for dt in px.index:
  x=sig.loc[dt]; y=fw.loc[dt]; ok=x.notna()&y.notna()
  if ok.sum()>=8: out.append(spearmanr(x[ok],y[ok]).statistic)
 print('decay',h,np.nanmean(out),len(out))
for n in [365,750,1260]:
 q=ic[-min(n,len(ic)):]; print('recent',n,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252),'dates',len(q))
out=pd.DataFrame({'date':sig.index}); [out.__setitem__(a,sig[a].values) for a in assets]; out.to_csv('scripts/miner_3_20350524_dispersion_gated_residual_reversal_signal.csv',index=False)
pd.DataFrame(rows,columns=['date','ic','n']).to_csv('scripts/miner_3_20350524_dispersion_gated_residual_reversal_ic.csv',index=False)
