import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
root='../persistent'
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(path):
 d=pd.read_csv(path); d['date']=pd.to_datetime(d['date']); return d.set_index('date').sort_index()
px={a:load(f'{root}/stock_data/{a}.csv') for a in assets}
dxy=load(f'{root}/index_data/DXY.csv')
# candidate: negative rolling DXY beta, 60 days, predict next asset return
rows=[]
for a in assets:
 d=px[a].join(dxy[['close']].rename(columns={'close':'dxy'}),how='inner').copy()
 d['ra']=d.close.pct_change(); d['rd']=d.dxy.pct_change()
 cov=d.ra.rolling(60,min_periods=45).cov(d.rd); var=d.rd.rolling(60,min_periods=45).var()
 d['f']=-(cov/var)
 d['fwd']=d.close.shift(-1)/d.close-1
 d['asset']=a; rows.append(d[['f','fwd','asset']].reset_index())
x=pd.concat(rows).dropna(); obs=[]
for dt,g in x.groupby('date'):
 if len(g)>=8 and g.f.nunique()>1 and g.fwd.nunique()>1: obs.append((dt,spearmanr(g.f,g.fwd).statistic,len(g)))
o=pd.DataFrame(obs,columns=['date','ic','n'])
print('candidate negative DXY beta; dates',len(o),'avg_n',o.n.mean(),'coverage',len(x)/(len(assets)*len(set(x.date))))
print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(ddof=1),'hit',(o.ic>0).mean(),'std',o.ic.std(ddof=1))
for y in [2020,2021,2022,2023,2024,2025,2026]:
 z=o[o.date.dt.year==y]; print(y,len(z),z.ic.mean(),z.ic.mean()/z.ic.std(ddof=1) if len(z)>2 else np.nan)
for h in [5,10]:
 rows=[]
 for a in assets:
  d=px[a].join(dxy[['close']].rename(columns={'close':'dxy'}),how='inner'); d['ra']=d.close.pct_change();d['rd']=d.dxy.pct_change(); d['f']=-(d.ra.rolling(60,min_periods=45).cov(d.rd)/d.rd.rolling(60,min_periods=45).var());d['fwd']=d.close.shift(-h)/d.close-1;d['asset']=a;rows.append(d[['f','fwd','asset']].reset_index())
 z=pd.concat(rows).dropna(); oo=[]
 for dt,g in z.groupby('date'):
  if len(g)>=8: oo.append(spearmanr(g.f,g.fwd).statistic)
 print(h,len(oo),np.mean(oo),np.mean(oo)/np.std(oo,ddof=1))
print('turnover proxy', x.sort_values(['asset','date']).groupby('date').f.apply(lambda z: z.rank(pct=True).mean()).diff().abs().mean())
