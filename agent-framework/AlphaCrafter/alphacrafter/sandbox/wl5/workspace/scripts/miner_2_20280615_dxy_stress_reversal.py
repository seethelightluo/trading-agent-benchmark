import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
end=pd.Timestamp('2028-06-15')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in assets},axis=1).sort_index()
macro=pd.read_csv('../persistent/index_data/DXY.csv'); macro.date=pd.to_datetime(macro.date); macro=macro.set_index('date').close
idx=px.index.intersection(macro.index); px=px.loc[idx]; macro=macro.loc[idx]
# signal: contrarian 5d return, active when DXY has positive 20d trend (stress proxy)
dxy20=macro.pct_change(20)
rows=[]; signals=[]
for i in range(30,len(px)-10):
 dt=px.index[i]
 if dt>end: break
 if not np.isfinite(dxy20.iloc[i]) or dxy20.iloc[i]<=0: continue
 r5=px.iloc[i]/px.iloc[i-5]-1
 fwd=px.iloc[i+10]/px.iloc[i]-1
 z=pd.Series(r5,index=assets).replace([np.inf,-np.inf],np.nan)
 valid=z.notna() & fwd.notna()
 if valid.sum()>=8:
  ic=spearmanr((-z[valid]).values,fwd[valid].values).statistic
  rows.append((dt,ic)); signals.append((-z).rank(pct=True))
r=pd.Series(dict(rows)).sort_index(); print('dates',len(r),'instruments',len(assets),'meanN',len(assets),'coverage',len(r)/(len(px)-40))
print('IC %.6f ICIR %.6f hit %.4f'%(r.mean(),r.mean()/r.std(ddof=1), (r>0).mean()))
# regimes
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-06-15')]:
 q=r.loc[a:b]; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan,(q>0).mean())
# rank turnover consecutive active observations
ss=pd.DataFrame(signals,index=r.index).fillna(0); print('turnover',ss.diff().abs().mean().mean())
for h in [1,5,10,20]:
 vals=[]
 for i in range(30,len(px)-h):
  dt=px.index[i]
  if dt>end or dxy20.iloc[i]<=0: continue
  z=px.iloc[i]/px.iloc[i-5]-1; fw=px.iloc[i+h]/px.iloc[i]-1; v=z.notna()&fw.notna()
  if v.sum()>=8: vals.append(spearmanr((-z[v]).values,fw[v].values).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
# artifact signal dates/factor values for audit
out=[]
for dt,s in zip(r.index,signals):
 for a,v in s.items(): out.append({'date':dt,'asset':a,'signal':float(v) if pd.notna(v) else np.nan})
pd.DataFrame(out).to_csv('scripts/miner_2_20280615_dxy_stress_reversal_signal.csv',index=False)
