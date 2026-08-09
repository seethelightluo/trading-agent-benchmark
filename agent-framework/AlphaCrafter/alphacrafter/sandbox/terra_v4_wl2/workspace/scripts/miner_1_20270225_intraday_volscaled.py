import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2027-02-25')
rows=[]; sig={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date'); d=d[d.date<=cut].set_index('date')
 intr=d.close/d.open-1
 vol=d.close.pct_change().rolling(20,min_periods=10).std()
 f=-intr/vol.replace(0,np.nan)
 r=d.close.pct_change().shift(-1)
 sig[s]=f
 for dt in d.index:
  if pd.notna(f.loc[dt]) and pd.notna(r.loc[dt]): rows.append((dt,s,f.loc[dt],r.loc[dt]))
a=pd.DataFrame(rows,columns=['date','s','f','r']); out=[]
for dt,g in a.groupby('date'):
 g=g.replace([np.inf,-np.inf],np.nan).dropna()
 if len(g)>=8:
  q=spearmanr(g.f,g.r).statistic
  if np.isfinite(q): out.append((dt,q,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('dates',len(z),'avg_n',z.n.mean(),'coverage',len(z)/a.date.nunique(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
 q=z.loc[lo:hi].ic; print(lo+'-'+hi,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# rank turnover: average fraction changing top/bottom rank ordering
ranks=pd.DataFrame({s:pd.Series(v).rank(pct=True) for s,v in sig.items()}).sort_index(); common=ranks.dropna(how='all')
turn=[]
for i in range(1,len(common)):
 a1=common.iloc[i-1]; a2=common.iloc[i]; ix=a1.notna()&a2.notna();
 if ix.sum()>=8: turn.append(np.mean(np.sign(a1[ix]-0.5)!=np.sign(a2[ix]-0.5)))
print('median_signal_turnover_proxy',np.nanmean(turn),'valid_dates',len(common))
pd.DataFrame(sig).to_csv('../persistent/factor_signals_miner_1_20270225_intraday_volscaled.csv')
