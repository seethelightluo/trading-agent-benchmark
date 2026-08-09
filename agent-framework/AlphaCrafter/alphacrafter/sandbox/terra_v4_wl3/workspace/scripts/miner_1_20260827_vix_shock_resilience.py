import pandas as pd, numpy as np
from scipy.stats import spearmanr
base='../persistent'; syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv(f'{base}/stock_data/{s}.csv',parse_dates=['date']).set_index('date').close for s in syms}
vix=pd.read_csv(f'{base}/index_data/VIX.csv',parse_dates=['date']).set_index('date').close
r=pd.DataFrame(px).pct_change(); vr=vix.pct_change(); rows=[]
for i,d in enumerate(r.index):
 if d not in vr.index or i+10>=len(r): continue
 hist=r.iloc[:i+1].tail(120); vh=vr.reindex(hist.index)
 if len(hist)<80 or vh.notna().sum()<60: continue
 shock=vh>=vh.quantile(.80)
 if shock.sum()<8: continue
 f=hist.loc[shock].mean()*1e3 + .25*hist.tail(20).mean()*1e3
 for s in syms:
  if pd.notna(f.get(s)): rows.append((d,s,f[s]))
df=pd.DataFrame(rows,columns=['date','sym','factor'])
def run(h):
 vals=[]
 for d,g in df.groupby('date'):
  i=r.index.get_loc(d); y=r.iloc[i+1:i+1+h].sum().reindex(g.sym).values
  z=pd.DataFrame({'x':g.set_index('sym').factor,'y':y},index=g.sym).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic)
 a=np.array(vals); return len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),(a>0).mean()
for h in [1,5,10]: print(h,*(round(x,5) if isinstance(x,float) else x for x in run(h)))
print('dates',df.date.nunique(),'avg names',round(df.groupby('date').size().mean(),2),'coverage',round(len(df)/(r.shape[0]*15),4))
for yr,g in df.groupby(df.date.dt.year):
 vals=[]
 for d,z in g.groupby('date'):
  i=r.index.get_loc(d); y=r.iloc[i+1:i+2].sum().reindex(z.sym).values
  zz=pd.DataFrame({'x':z.set_index('sym').factor,'y':y}).dropna()
  if len(zz)>=8: vals.append(spearmanr(zz.x,zz.y).statistic)
 print(yr,len(vals),round(np.nanmean(vals),4),round(np.nanmean(vals)/np.nanstd(vals,ddof=1),4))
