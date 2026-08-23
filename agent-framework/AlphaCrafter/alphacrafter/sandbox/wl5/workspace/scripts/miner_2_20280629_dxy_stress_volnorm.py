import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
end=pd.Timestamp('2028-06-29'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def ld(s):
 p='../persistent/index_data/'+s+'.csv';
 if not os.path.exists(p):p='../persistent/stock_data/'+s+'.csv'
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);return d.set_index('date').close
px=pd.concat({s:ld(s) for s in A},axis=1).sort_index(); d=pd.read_csv('../persistent/index_data/DXY.csv');d.date=pd.to_datetime(d.date);m=d.set_index('date').close
ix=px.index.intersection(m.index);px=px.loc[ix];m=m.loc[ix]; active=m.pct_change(20)>0
rows=[]; out=[]
for i in range(30,len(px)-10):
 if px.index[i]>end or not active.iloc[i]:continue
 r=px.iloc[i]/px.iloc[i-20]-1; vol=px.pct_change().iloc[i-20:i].std()*np.sqrt(20); z=r/vol
 fw=px.iloc[i+10]/px.iloc[i]-1;v=z.notna()&fw.notna()
 if v.sum()>=8:
  rows.append((px.index[i],spearmanr((-z[v]).values,fw[v].values).statistic));out.append(pd.Series(-z,index=A).rank(pct=True))
r=pd.Series(dict(rows)).sort_index();print('dates',len(r),'instruments',15,'meanN',15,'coverage',len(r)/(len(px)-40));print('IC %.6f ICIR %.6f hit %.4f'%(r.mean(),r.mean()/r.std(ddof=1),(r>0).mean()))
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-06-29')]:
 q=r.loc[a:b];print('regime',a,len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
print('turnover',pd.DataFrame(out,index=r.index).diff().abs().mean().mean())
for h in [1,5,10,20]:
 z=[]
 for i in range(30,len(px)-h):
  if px.index[i]>end or not active.iloc[i]:continue
  rr=px.iloc[i]/px.iloc[i-20]-1;vv=px.pct_change().iloc[i-20:i].std()*np.sqrt(20); fw=px.iloc[i+h]/px.iloc[i]-1;x=rr/vv;v=x.notna()&fw.notna()
  if v.sum()>=8:z.append(spearmanr((-x[v]).values,fw[v].values).statistic)
 print('decay',h,'IC %.6f n %d'%(np.nanmean(z),len(z)))
pd.DataFrame([{'date':dt,'asset':a,'signal':float(v)} for dt,s in zip(r.index,out) for a,v in s.items()]).to_csv('scripts/miner_2_20280629_dxy_stress_volnorm_signal.csv',index=False)
