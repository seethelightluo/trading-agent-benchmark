import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 r=d['close'].pct_change()
 # trend efficiency: directional displacement divided by path length, signed
 D[s]=pd.DataFrame({'f':r.rolling(20,min_periods=18).sum()/r.abs().rolling(20,min_periods=18).sum(),'f5':r.rolling(5,min_periods=5).sum()/r.abs().rolling(5,min_periods=5).sum(),'r':d.close.pct_change()})
# factor at t predicts next close return, exact per asset
rows=[]
for s,x in D.items():
 z=x.copy(); z['y']=x['r'].shift(-1); z['s']=s; rows.append(z)
a=pd.concat(rows).reset_index().dropna(subset=['f','y'])
ics=[]
for dt,g in a.groupby('date'):
 if len(g)>=8: ics.append((dt,spearmanr(g.f,g.y).statistic,len(g)))
v=np.array([q[1] for q in ics]); print('dates',len(v),'avg_n',np.mean([q[2] for q in ics]),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean())
# alternatives
for col in ['f5']:
 vv=[]
 for dt,g in a.dropna(subset=[col]).groupby('date'):
  if len(g)>=8: vv.append(spearmanr(g[col],g.y).statistic)
 vv=np.array(vv); print(col,len(vv),vv.mean(),vv.mean()/vv.std(ddof=1))
for yr in [2020,2021,2022,2023,2024,2025,2026]:
 vv=[q[1] for q in ics if q[0].year==yr]; print(yr,len(vv),np.mean(vv) if vv else np.nan)
# turnover ranks
wide=a.pivot(index='date',columns='s',values='f'); ranks=wide.rank(axis=1,pct=True); print('coverage',a.f.notna().mean(),'turnover',ranks.diff().abs().mean(axis=1).mean())
