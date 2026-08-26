import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2031-01-10')
px={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
# trend persistence quality: 20-session signed momentum scaled by 10-session path efficiency
mom=P/P.shift(20)-1
r10=R.rolling(10).sum(); eff=r10.abs()/(R.abs().rolling(10).sum()+1e-12)
f=(mom*eff).clip(-3,3)
ics=[]; cover=[]; vals=[]
for dt in P.index:
 if dt not in f.index: continue
 nxt=P.shift(-1).loc[dt]/P.loc[dt]-1
 z=f.loc[dt]; ok=z.notna()&nxt.notna()
 if ok.sum()>=8:
  ic=spearmanr(z[ok],nxt[ok]).statistic
  if np.isfinite(ic): ics.append(ic); cover.append(ok.sum()); vals.append((dt,ic))
a=np.array(ics); print('factor trend20_eff10'); print('dates',len(a),'avgN',np.mean(cover),'coverage',np.mean(cover)/15)
print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'absmean',abs(a.mean()))
# turnover ranks
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean(); print('turnover',turn)
for h in [5,10,20]:
 fw=P.shift(-h)/P-1; q=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&fw.loc[dt].notna()
  if ok.sum()>=8:q.append(spearmanr(f.loc[dt][ok],fw.loc[dt][ok]).statistic)
 q=np.array(q);print('h',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'n',len(q))
# artifacts
out=pd.DataFrame(f); out.index.name='date'; out.to_csv('scripts/miner_1_20310113_trend20_eff10_signal.csv')
pd.DataFrame(vals,columns=['date','ic']).to_csv('scripts/miner_1_20310113_trend20_eff10_ic.csv',index=False)
