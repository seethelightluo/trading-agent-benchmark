import numpy as np,pandas as pd,glob,os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; paths={os.path.basename(x)[:-4]:x for x in glob.glob('../persistent/**/*.csv',recursive=True)}; end=pd.Timestamp('2026-07-15'); O={}; C={}
for s in U:
 d=pd.read_csv(paths[s]); d.date=pd.to_datetime(d.date); d=d[d.date<=end].drop_duplicates('date').set_index('date'); O[s]=d.open; C[s]=d.close
op=pd.DataFrame(O).sort_index(); cl=pd.DataFrame(C).sort_index(); r=cl.pct_change(); f=(op/cl-1)/r.rolling(20,min_periods=15).std(); fr=r.shift(-1); I=[];D=[];CV=[]
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:I.append(a[ok].corr(b[ok]));D.append(dt);CV.append(ok.mean())
s=pd.Series(I,index=D); print('dates',len(s),'avg_names',np.mean([((f.loc[d].notna())&(fr.loc[d].notna())).sum() for d in D]));print('IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean(),np.mean(CV)))
for y,g in s.groupby(lambda z:z.year):print(y,round(g.mean(),5),len(g))
