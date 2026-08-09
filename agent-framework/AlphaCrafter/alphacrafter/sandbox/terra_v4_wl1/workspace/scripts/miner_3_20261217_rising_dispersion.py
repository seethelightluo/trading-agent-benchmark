import pandas as pd, numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
allr=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END]; r=d.close.pct_change()
 allr.append(pd.DataFrame({'date':d.date,'symbol':s,'r3':d.close.pct_change(3),'vol':r.rolling(20,min_periods=10).std(),'y1':d.close.shift(-1)/d.close-1}))
x=pd.concat(allr,ignore_index=True); p=x.pivot(index='date',columns='symbol',values='r3'); disp=p.std(axis=1); rising=(disp>disp.rolling(20,min_periods=10).mean()).astype(float); rising[rising.index<disp.index[0]]=np.nan
# reversal only during rising dispersion, demean cross-section; zero outside regime would create ties, retain active dates only
med=p.median(axis=1); x['factor']=-(x.r3-x.date.map(med))/x.vol.replace(0,np.nan); x['regime']=x.date.map(rising); x.loc[x.regime<0.5,'factor']=np.nan
def calc(df):
 out=[];ns=[]
 for dt,g in df.groupby('date'):
  g=g.dropna(subset=['factor','y1'])
  if len(g)>=8: out.append(spearmanr(g.factor,g.y1).statistic);ns.append(len(g))
 a=np.array(out); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean()
print('UNIVERSE',len(syms),'rows',len(x));print('H1',calc(x))
for lo,hi,n in [('2020','2022','2020-22'),('2023','2024','2023-24'),('2025','2026-12-17','2025-26')]:print(n,calc(x[(x.date>=lo)&(x.date<=hi)]))
for h in [5,10]:
 ys=[]
 for s in syms:
  d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];ys.append(pd.DataFrame({'date':d.date,'symbol':s,'yf':d.close.shift(-h)/d.close-1}))
 z=x[['date','symbol','factor']].merge(pd.concat(ys),on=['date','symbol']).rename(columns={'yf':'y1'});print('H',h,calc(z))
v=x.dropna(subset=['factor']); print('coverage_active',len(v)/len(x),'active_dates',v.date.nunique(),'turnover',v.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());v[['date','symbol','factor']].to_csv('scripts/miner_3_20261217_rising_dispersion_signal.csv',index=False)
