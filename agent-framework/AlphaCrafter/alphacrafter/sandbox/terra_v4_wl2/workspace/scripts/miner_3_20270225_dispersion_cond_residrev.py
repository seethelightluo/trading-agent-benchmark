import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
r=pd.DataFrame({a:p[a].pct_change() for a in A}); r3=pd.DataFrame({a:p[a].pct_change(3) for a in A}); vol=r.rolling(20,min_periods=10).std(); resid=r3.sub(r3.median(axis=1),axis=0); raw=-resid.div(vol,axis=0)
disp=r3.std(axis=1); threshold=disp.rolling(60,min_periods=30).median()
rows=[]; sig=[]
for dt in r.index:
    vals=raw.loc[dt]; med=vals.median(); active=np.isfinite(threshold.get(dt,np.nan)) and disp.get(dt,np.nan)>threshold.get(dt,np.nan)
    for a in A: sig.append((dt,a,(vals[a]-med) if active and np.isfinite(vals[a]) else np.nan))
    if not active: continue
    for h in [1,5]:
      f=[];y=[]
      for a in A:
       if not np.isfinite(vals[a]) or dt not in p[a].index: continue
       ix=p[a].index.get_loc(dt)
       if ix+h<len(p[a]): f.append(vals[a]-med);y.append(p[a].iloc[ix+h]/p[a].iloc[ix]-1)
      if len(f)>=8: rows.append((dt,h,spearmanr(f,y).statistic,len(f)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5]:
 z=d[d.h==h];print('H',h,'dates',len(z),'avg_n',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(),'hit',(z.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  q=z.set_index('date').loc[lo:hi].ic;print(lo,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
out=pd.DataFrame(sig,columns=['date','asset','signal']);out.to_csv('../persistent/factor_signals_miner_3_20270225_dispersion_cond_residrev.csv',index=False);print('artifact',len(out),'active',out.signal.notna().groupby(out.date).any().sum())
print('turnover',out.pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
