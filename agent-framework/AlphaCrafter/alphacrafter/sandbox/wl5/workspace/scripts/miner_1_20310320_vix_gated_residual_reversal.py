import numpy as np, pandas as pd, os
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date)
 C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
# Observation-only VIX percentile, lagged naturally by using date i value (decision after close i).
vix=pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date'])
vc=vix.set_index('date')['close'].astype(float).reindex(P.index).ffill()
# 5d cross-sectional residual reversal, amplified in high-volatility regimes.
rows=[]; sig=[]
for i in range(120,len(P)-5):
 r5=R.iloc[i-4:i+1].sum(); resid=r5-r5.median()
 rv=R.iloc[i-59:i+1].std()+1e-12
 base=-resid/rv
 hist=vc.iloc[max(0,i-59):i+1].dropna()
 if len(hist)<30: continue
 pct=(hist.rank(pct=True).iloc[-1]); gate=1.0+0.75*max(0.0,pct-0.60)/0.40
 f=base*gate
 y=P.iloc[i+5]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 sig += [{'date':str(P.index[i].date()),'symbol':s,'signal':float(a)} for s,a in f.items()]
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2031-03-19')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6) if len(q) else None)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
pd.DataFrame(sig).to_csv('scripts/miner_1_20310320_vix_gated_residual_reversal_5d_signal.csv',index=False)
