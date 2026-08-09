import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17')
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
 r=d.close.pct_change(); rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r7':d.close.pct_change(7),'vol20':r.rolling(20,min_periods=10).std(),'y1':d.close.shift(-1)/d.close-1,'y5':d.close.shift(-5)/d.close-1,'y10':d.close.shift(-10)/d.close-1}))
x=pd.concat(rows,ignore_index=True)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date');vix=vix[vix.date<=END].set_index('date').close
# Use only information available at t: lagged VIX and rolling historical median.
reg=pd.DataFrame({'vix':vix.shift(1)});reg['med60']=reg.vix.rolling(60,min_periods=30).median();reg['high']=(reg.vix>reg.med60).astype(float)
w=x.pivot(index='date',columns='symbol',values='r7'); med=w.median(axis=1); med[w.count(axis=1)<8]=np.nan
x['base']=-(x.r7-x.date.map(med))/x.vol20.replace(0,np.nan)
x=x.join(reg,on='date'); x['factor']=x.base*x.high

def calc(df,col):
 a=[];ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=[col,'y'])
  if len(g)>=8 and g[col].nunique()>1:a.append(spearmanr(g[col],g.y).statistic);ns.append(len(g))
 a=np.array(a);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),float((a>0).mean())
print('UNIVERSE',len(syms),'rows',len(x),'high_regime_share',x.groupby('date').high.first().mean())
for c in ['y1','y5','y10']:
 z=x.rename(columns={c:'y'});print(c,calc(z,'factor'))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)].rename(columns={'y1':'y'}),'factor'))
v=x.dropna(subset=['factor']); ranks=v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('coverage',len(v)/len(x),'turnover',ranks.diff().abs().mean(axis=1).mean())
v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_vix_gated_residual_signal.csv',index=False)
