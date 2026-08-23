import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; px={}; cut=pd.Timestamp('2028-10-05')
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d.date=pd.to_datetime(d.date); d=d[d.date<=cut]
 px[s]=d.sort_values('date').set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index(); r=np.log(P).diff(); vol=r.rolling(20).std(); m20=r.rolling(20).sum(); m5=r.rolling(5).sum(); down=r.where(r<0,0).rolling(20).std()
f=(m20/(vol*np.sqrt(20))).where(m20*m5>0,0.0)/(1+2*down*np.sqrt(20)); f=f.replace([np.inf,-np.inf],np.nan)
for h in [5,10,15,20]:
 fr=np.log(P.shift(-h)/P); vals=[]
 for dt in P.index:
  ok=f.loc[dt].notna()&fr.loc[dt].notna()
  if ok.sum()>=8 and f.loc[dt,ok].nunique()>1: vals.append(spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic)
 vals=pd.Series(vals); print('horizon',h,'dates',len(vals),'IC %.6f ICIR %.6f hit %.4f'%(vals.mean(),vals.mean()/vals.std(),(vals>0).mean()))
fr=np.log(P.shift(-10)/P); rows=[]
for dt in P.index:
 ok=f.loc[dt].notna()&fr.loc[dt].notna()
 if ok.sum()>=8 and f.loc[dt,ok].nunique()>1: rows.append((dt,spearmanr(f.loc[dt,ok],fr.loc[dt,ok]).statistic,ok.sum()))
R=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');
for lab,m in [('2020-24',R.index<'2025-01-01'),('2025-26',(R.index>='2025-01-01')&(R.index<'2027-01-01')),('2027+',R.index>='2027-01-01'),('2028+',R.index>='2028-01-01')]:
 z=R.loc[m].ic; print(lab,'dates',len(z),'IC %.6f ICIR %.6f hit %.4f'%(z.mean(),z.mean()/z.std(),(z>0).mean()))
print('range',R.index.min().date(),R.index.max().date(),'avg_n',R.n.mean(),'coverage',R.n.mean()/15)
rank=f.loc[R.index].rank(pct=True); print('turnover',rank.diff().abs().mean(axis=1).mean())
f.loc[R.index].stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20281005_confirmed_trend_signal.csv',index=False)
