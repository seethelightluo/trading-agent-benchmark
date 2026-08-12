import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d.close.astype(float) for s,d in D.items()}).ffill()
# Drawdown recovery: assets that have recovered from a medium-term trough,
# but remain below their prior peak, favoring orderly recovery over fresh extremes.
roll_hi=P.rolling(60,min_periods=40).max(); roll_lo=P.rolling(60,min_periods=40).min()
range_pos=(P-roll_lo)/(roll_hi-roll_lo).replace(0,np.nan)
recovery=(P/P.shift(10)-1) - (P/P.shift(40)-1)/4
# penalize current drawdown and normalize by recent volatility
from_peak=P/roll_hi-1
vol=P.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252)
f=((0.7*range_pos+0.3*recovery.rank(axis=1,pct=True)) * (1+from_peak.clip(-.5,0))).div(vol,axis=0).shift(1)
fr=np.log(P.shift(-10)/P); rows=[]
for dt in f.index:
 a,b=f.loc[dt],fr.loc[dt];ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append((dt,a[ok].corr(b[ok]),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=z.ic
print('dates',len(z),'avgN',z.n.mean(),'assets',len(U),'coverage',z.n.mean()/len(U),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'turnover',f.rank(pct=True).diff().abs().mean(axis=1).mean())
for n in [120,252,756]:
 x=q.tail(n);print('recent',n,len(x),x.mean(),x.mean()/x.std(ddof=1))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 x=q.loc[a:b];print(a,b,len(x),x.mean(),x.mean()/x.std(ddof=1))
f.to_csv('scripts/miner_3_20320916_drawdown_recovery_signal.csv');z.to_csv('scripts/miner_3_20320916_drawdown_recovery_ic.csv')
print('signal_path scripts/miner_3_20320916_drawdown_recovery_signal.csv')
print('ic_path scripts/miner_3_20320916_drawdown_recovery_ic.csv')
print('library_corr null (requires deterministic post-Miner recomputation)')
