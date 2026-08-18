import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-12-17')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}).sort_index()
p=p.loc[p.index<=CUT]; r=p.pct_change()
vx=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].reindex(p.index).ffill().pct_change()
shock=vx.gt(0)
f=pd.DataFrame(index=p.index,columns=U,dtype=float)
for i,dt in enumerate(p.index):
 if i<60: continue
 h=r.iloc[i-60:i]; sh=shock.iloc[i-60:i]
 if sh.sum()>=5: f.loc[dt]=-(h.where(sh,axis=0).mean()-h.mean())

def run(h):
 fw=p.pct_change(h).shift(-h); rows=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1: rows.append((dt,spearmanr(a.f,a.y).statistic,len(a)))
 z=np.array([x[1] for x in rows]); print('h',h,'dates',len(z),'meanN',round(np.mean([x[2] for x in rows]),2),'IC',round(z.mean(),8),'ICIR',round(z.mean()/z.std(ddof=1),8),'hit',round((z>0).mean(),6))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027)]:
  q=np.array([x[1] for x in rows if lo<=x[0].year<=hi]); print('regime',lo,hi,'mean',round(q.mean(),8) if len(q) else None,'n',len(q))
 return z
z=run(10)
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'instruments',len(U),'period',p.index.min().date(),p.index.max().date())
# save recoverable signal artifact through cutoff
out=f.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_2_20271217_vix_shock_resilience_signal.csv',index=False)
