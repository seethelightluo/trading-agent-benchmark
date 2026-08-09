import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'] for a in U}
px=pd.DataFrame(P).sort_index(); r=px.pct_change();
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close']
vr=v.rolling(252,min_periods=60).rank(pct=True).shift(1).reindex(px.index)
bread=(r.lt(0).sum(axis=1)/r.notna().sum(axis=1)).shift(1)
# lagged stress intensity; reversal only in objectively stressed breadth/VIX regimes
stress=((vr>.70)|(bread>.60)).astype(float)*(0.5+0.5*vr.fillna(.5))
raw=-r.rolling(3).sum(); F=raw.mul(stress,axis=0)
fr=px.shift(-5)/px-1
rows=[]; sig=[]
for d in px.index:
 z=F.loc[d]; y=fr.loc[d]; q=pd.concat([z,y],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:
  ic=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic; rows.append((d,ic,len(q)))
 med=z.median()
 for a in U: sig.append((d,a,z[a]-med if np.isfinite(z[a]) and np.isfinite(med) else np.nan))
x=pd.DataFrame(rows,columns=['date','ic','n']); x=x.replace([np.inf,-np.inf],np.nan).dropna()
print('dates',len(x),'avgN',round(x.n.mean(),2),'IC',round(x.ic.mean(),8),'ICIR',round(x.ic.mean()/x.ic.std(ddof=1),8),'hit',round((x.ic>0).mean(),6),'coverage',round(F.notna().sum().sum()/(15*len(F)),6),'active_frac',round((stress>0).mean(),6))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
 q=x.set_index('date').loc[lo:hi].ic; print(lo+'_'+hi,'dates',len(q),'IC',round(q.mean(),8) if len(q) else np.nan,'ICIR',round(q.mean()/q.std(ddof=1),8) if len(q)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_1_20270225_stress_breadth_reversal.csv',index=False)
