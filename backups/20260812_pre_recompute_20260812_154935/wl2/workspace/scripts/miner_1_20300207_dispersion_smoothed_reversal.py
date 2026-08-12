import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
xs={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<250: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d)>250:
  d=d.copy(); d['date']=pd.to_datetime(d['date']); xs[s]=d.set_index('date')['close'].astype(float).pct_change()
r=pd.DataFrame(xs).sort_index(); mom=r.rolling(20,min_periods=15).sum(); csdisp=mom.std(axis=1)
raw=mom.sub(mom.mean(axis=1),axis=0); mult=(csdisp/csdisp.rolling(60,min_periods=20).median()).clip(0.5,2.0)
f=-raw.mul(mult,axis=0).ewm(span=5,min_periods=3).mean()
rows=[]; turnover=[]
for i in range(len(r)-10):
 a=f.iloc[i]; fr=r.iloc[i+1:i+6].sum(); ok=a.notna()&fr.notna()
 if ok.sum()>=8: rows.append((r.index[i],ok.sum(),a[ok].corr(fr[ok]),a[ok].corr(r.iloc[i+1][ok])))
 if i>0: turnover.append((f.iloc[i].rank(pct=True)-f.iloc[i-1].rank(pct=True)).abs().mean())
z=pd.DataFrame(rows,columns=['date','n','ic5','ic1']).set_index('date')
def stat(x):
 x=x.dropna(); return (len(x),round(x.mean(),5),round(x.std(ddof=1),5),round(x.mean()/x.std(ddof=1),5),round((x>0).mean(),4))
print('dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.sum()/(len(z)*15),4),'turnover',round(np.mean(turnover),4))
for c in ['ic1','ic5']:
 print(c,stat(z[c]))
 for label,sub in [('2020-22',z.loc[:'2022-12-31']),('2023-25',z.loc['2023-01-01':'2025-12-31']),('2026-28',z.loc['2026-01-01':'2028-12-31']),('2029+',z.loc['2029-01-01':])]: print(label,stat(sub[c]))
f.to_csv('scripts/miner_1_20300207_dispersion_smoothed_reversal_signal.csv',index_label='date')
