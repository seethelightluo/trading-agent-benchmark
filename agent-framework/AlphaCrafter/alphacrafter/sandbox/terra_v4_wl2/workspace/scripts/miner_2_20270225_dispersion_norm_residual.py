import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
P=pd.DataFrame(p); r=P.pct_change(); med=r.median(axis=1)
# Candidate: residual 3-day reversal, risk-normalized, activated smoothly by cross-asset dispersion surprise.
res=(r.sub(med,axis=0)).rolling(3,min_periods=3).sum()
vol=r.rolling(20,min_periods=10).std(); disp=r.sub(med,axis=0).abs().mean(axis=1)
activation=((disp/disp.rolling(60,min_periods=30).median())-1).clip(0,2)
raw=-res.div(vol).mul(activation,axis=0)
rows=[]; sig=[]
for dt in P.index:
 vals=raw.loc[dt]; good=vals.dropna(); center=good.median() if len(good)>=8 else np.nan
 for a in A: sig.append((dt,a, vals[a]-center if np.isfinite(center) and np.isfinite(vals[a]) else np.nan))
 for h in [1,5,10]:
  y=P.shift(-h).loc[dt]/P.loc[dt]-1; z=(vals-center).dropna(); q=pd.concat([z.rename('f'),y.rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: rows.append((dt,h,spearmanr(q.f,q.y).statistic,len(q)))
d=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=d[d.h==h]; print('H',h,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  x=q.set_index('date').loc[lo:hi].ic; print(lo,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else None)
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_2_20270225_dispersion_norm_residual.csv',index=False)
w=out.pivot(index='date',columns='asset',values='signal'); print('artifact',len(out),'coverage',round(out.signal.notna().mean(),4),'turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
