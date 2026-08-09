import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date'); px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); vol=r.rolling(20).std()
accel=vol.mean(axis=1)/vol.mean(axis=1).rolling(60).median()
csdisp=r.rolling(3).std().mean(axis=1)
active=(csdisp>csdisp.rolling(60).median())&(accel>1.10)
ret3=P.pct_change(3); resid=ret3.sub(ret3.median(axis=1),axis=0); f=(-resid/vol).where(active,np.nan); fr=P.pct_change(1).shift(-1)
def calc(y):
 rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
out=calc(fr); print('dates',len(out),'active',int(active.sum()),'avg_n',out.n.mean(),'coverage',float(f.notna().sum().sum()/f.size)); print('IC %.9f ICIR %.9f hit %.5f'%(out.ic.mean(),out.ic.mean()/out.ic.std(ddof=1),(out.ic>0).mean()))
for h in [1,3,5]:
 q=calc(P.pct_change(h).shift(-h)); print('h',h,'ic',q.ic.mean(),'n',len(q))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027-02-25')]:
 q=out.loc[a:b].ic; print(a,b,len(q),q.mean() if len(q) else np.nan)
rank=f.rank(axis=1,pct=True); common=rank.notna().sum(axis=1)>=8; turn=[]
for i in range(1,len(rank)):
 if common.iloc[i] and common.iloc[i-1]: turn.append((rank.iloc[i]-rank.iloc[i-1]).abs().mean())
print('active frac',active.mean(),'turn',np.nanmean(turn)); f.to_csv('../persistent/factor_signals_miner_3_20270225_volaccel_disp_reversal.csv')
