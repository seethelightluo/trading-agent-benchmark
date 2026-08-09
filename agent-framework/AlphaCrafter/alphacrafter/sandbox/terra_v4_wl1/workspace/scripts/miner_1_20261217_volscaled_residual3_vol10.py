import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]; r=d.close.pct_change(); z.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'v10':r.rolling(10,min_periods=5).std(),'y':d.close.shift(-1)/d.close-1}))
x=pd.concat(z); w=x.pivot(index='date',columns='symbol',values='r3'); m=w.median(axis=1);m[w.count(axis=1)<8]=np.nan;x['f']=-(x.r3-x.date.map(m))/x.v10.replace(0,np.nan)
def c(q):
 a=[];n=[]
 for _,g in q.groupby('date'):
  g=g.dropna(subset=['f','y'])
  if len(g)>=8:a+=[spearmanr(g.f,g.y).statistic];n+=[len(g)]
 a=np.array(a);return len(a),np.mean(n),np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0)
print('universe',len(syms),'rows',len(x),'all',c(x))
for lo,hi,nm in [('2020','2022','20-22'),('2023','2024','23-24'),('2025','2026','25-26')]:print(nm,c(x[(x.date>=lo)&(x.date<=hi)]))
print('coverage',x.f.notna().mean(),'turnover',x.dropna().pivot(index='date',columns='symbol',values='f').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
x[['date','symbol','f']].dropna().rename(columns={'f':'factor'}).to_csv('scripts/miner_1_20261217_volscaled_residual3_vol10_signal.csv',index=False)
