import numpy as np, pandas as pd, glob, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
files=glob.glob('../persistent/**/*.csv',recursive=True)
print('files',len(files),files[:5])
paths={os.path.basename(x)[:-4]:x for x in files}
frames={}
for s in U:
 p=paths.get(s)
 if not p: print('missing',s); continue
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); frames[s]=d.drop_duplicates('date').set_index('date')['close']
macro=pd.read_csv(paths['DXY']); macro['date']=pd.to_datetime(macro['date']); macro=macro.drop_duplicates('date').set_index('date')['close']
px=pd.DataFrame(frames).sort_index(); common=px.join(macro.rename('DXY'),how='inner'); r=common.pct_change(); x=r.DXY
f=-r[U].rolling(60,min_periods=45).cov(x).div(x.rolling(60,min_periods=45).var(),axis=0); fr=px[U].pct_change().shift(-1)
ics=[]; dates=[]; cov=[]; turns=[]; prev=None
for dt in f.index:
 a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8:
  ics.append(a[ok].corr(b[ok])); dates.append(dt); cov.append(ok.mean())
  if prev is not None:
   q=prev.notna()&a.notna(); turns.append((a[q].rank(pct=True)-prev[q].rank(pct=True)).abs().mean())
  prev=a
ics=np.array(ics); ser=pd.Series(ics,index=dates)
print('dates',len(ics),'avg_names',np.mean([((f.loc[dt].notna())&(fr.loc[dt].notna())).sum() for dt in dates]))
print('IC %.6f ICIR %.6f hit %.4f coverage %.4f turnover %.4f'%(np.nanmean(ics),np.nanmean(ics)/np.nanstd(ics,ddof=1),np.mean(ics>0),np.mean(cov),np.nanmean(turns)))
for y,g in ser.groupby(lambda z:z.year): print(y,round(g.mean(),5),len(g))
print('period',common.index.min(),common.index.max())
