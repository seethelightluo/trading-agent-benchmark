import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
end=pd.Timestamp('2028-06-29')
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 p='../persistent/index_data/'+s+'.csv'
 if not os.path.exists(p): p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
px=pd.concat({s:load(s) for s in assets},axis=1).sort_index()
m=pd.read_csv('../persistent/index_data/DXY.csv'); m.date=pd.to_datetime(m.date); m=m.set_index('date').close
idx=px.index.intersection(m.index); px=px.loc[idx]; m=m.loc[idx]
stress=m.pct_change(20)>0
rows=[]; sig=[]
for i in range(30,len(px)-10):
 dt=px.index[i]
 if dt>end or not stress.iloc[i]: continue
 r20=px.iloc[i]/px.iloc[i-20]-1; fw=px.iloc[i+10]/px.iloc[i]-1
 v=r20.notna()&fw.notna()
 if v.sum()>=8:
  x=-r20[v]; y=fw[v]; rows.append((dt,spearmanr(x,y).statistic)); sig.append(pd.Series(-r20,index=assets).rank(pct=True))
r=pd.Series(dict(rows)).sort_index(); print('dates',len(r),'instruments',len(assets),'meanN',len(assets),'coverage',len(r)/(len(px)-40))
print('IC %.6f ICIR %.6f hit %.4f'%(r.mean(),r.mean()/r.std(ddof=1),(r>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-06-29')]:
 q=r.loc[a:b]; print('regime',a,b,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
ss=pd.DataFrame(sig,index=r.index); print('turnover',ss.diff().abs().mean().mean())
for h in [1,5,10,20]:
 z=[]
 for i in range(30,len(px)-h):
  if px.index[i]>end or not stress.iloc[i]: continue
  rr=px.iloc[i]/px.iloc[i-20]-1; fw=px.iloc[i+h]/px.iloc[i]-1; v=rr.notna()&fw.notna()
  if v.sum()>=8:z.append(spearmanr((-rr[v]).values,fw[v].values).statistic)
 print('decay',h,'IC %.6f n %d'%(np.nanmean(z),len(z)))
out=[]
for dt,s in zip(r.index,sig):
 for a,v in s.items(): out.append({'date':dt,'asset':a,'signal':float(v) if pd.notna(v) else np.nan})
pd.DataFrame(out).to_csv('scripts/miner_2_20280629_dxy_stress_medium_reversal_signal.csv',index=False)
