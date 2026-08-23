import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff()
v=get_index_daily_data('VIX',days=4000); v.date=pd.to_datetime(v.date); V=v.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]; sig=[]
# Volatility-regime switch: normalized 20d trend in calm regimes, its inverse in stressed regimes.
for i in range(80,len(P)-10):
 trend=P.iloc[i]/P.iloc[i-20]-1; vol=R.iloc[i-59:i+1].std()*np.sqrt(60); base=trend/(vol+1e-12)
 vp=V.iloc[i]; hist=V.iloc[max(0,i-120):i].dropna(); stressed=bool(len(hist)>30 and vp>hist.quantile(.6))
 f=(-base if stressed else base); f=f-f.median(); y=P.iloc[i+10]/P.iloc[i]-1
 z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,a in f.items(): sig.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(a)})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('candidate=macro_regime_switch_trend'); print('assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-10-02')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6) if len(z)>1 else np.nan)
S=pd.DataFrame(sig).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,20]:
 q=[]
 for i in range(80,len(P)-h):
  trend=P.iloc[i]/P.iloc[i-20]-1; vol=R.iloc[i-59:i+1].std()*np.sqrt(60); base=trend/(vol+1e-12); hist=V.iloc[max(0,i-120):i].dropna(); f=(-base if len(hist)>30 and V.iloc[i]>hist.quantile(.6) else base); y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1:q.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'dates',len(q),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20301017_macro_regime_switch_signal.csv',index=False)
