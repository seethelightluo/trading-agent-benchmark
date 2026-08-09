import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); common=r.median(axis=1); resid=r.sub(common,axis=0); vol=r.rolling(20,min_periods=10).std(); raw=-resid.div(vol,axis=0)
disp=r.std(axis=1); gate=disp>disp.rolling(60,min_periods=30).median()
rows=[]; sig=[]
for dt in raw.index:
 vals=raw.loc[dt]; med=vals.median() if vals.notna().sum()>=8 else np.nan
 active=bool(gate.get(dt,False))
 for a in A: sig.append((dt,a,(vals[a]-med) if active and pd.notna(vals[a]) and pd.notna(med) else np.nan))
 f=[];y=[]
 for a in A:
  if not active or dt not in p[a].index: continue
  i=p[a].index.get_loc(dt); z=vals[a]-med if pd.notna(vals[a]) and pd.notna(med) else np.nan
  if np.isfinite(z) and i+1<len(p[a]): f.append(z); y.append(p[a].iloc[i+1]/p[a].iloc[i]-1)
 if len(f)>=8: rows.append((dt,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','ic','n']); print('active_dates',len(d),'active_frac',len(d)/len(raw),'avg_n',d.n.mean(),'IC',d.ic.mean(),'ICIR',d.ic.mean()/d.ic.std(),'hit',(d.ic>0).mean())
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=d.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_conditional_dispersion_residual.csv',index=False); print('coverage',out.signal.notna().mean(),'turn',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
print('max_abs_library_correlation',None)
