import numpy as np,pandas as pd,glob,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; paths={os.path.basename(x)[:-4]:x for x in glob.glob('../persistent/**/*.csv',recursive=True)}; end=pd.Timestamp('2026-07-15')
O={}; C={}
for s in U:
 d=pd.read_csv(paths[s]); d.date=pd.to_datetime(d.date); d=d[d.date<=end].drop_duplicates('date').set_index('date'); O[s]=d['open']; C[s]=d.close
op=pd.DataFrame(O).sort_index(); cl=pd.DataFrame(C).sort_index(); intr=(op/cl-1); sig=intr.rolling(3,min_periods=3).mean(); fr=cl.pct_change().shift(-1)
ics=[]; dates=[]; cov=[]; turns=[]; prev=None
for dt in sig.index:
 a=sig.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(a[ok].corr(b[ok])); dates.append(dt); cov.append(ok.mean())
  if prev is not None:
   q=prev.notna()&a.notna(); turns.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
  prev=a
ser=pd.Series(ics,index=dates); print('dates',len(ser),'avg_names',np.mean([((sig.loc[d].notna())&(fr.loc[d].notna())).sum() for d in dates])); print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(ser.mean(),ser.mean()/ser.std(ddof=1),(ser>0).mean(),np.mean(cov),np.mean(turns)))
for y,g in ser.groupby(lambda z:z.year): print(y,round(g.mean(),5),len(g))
print('period',ser.index.min(),ser.index.max())
