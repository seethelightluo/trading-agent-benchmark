import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(P).sort_index(); R=np.log(P).diff()
v=get_index_daily_data('VIX',days=4000); v.date=pd.to_datetime(v.date); V=v.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float).reindex(P.index).ffill()
rows=[]; signals=[]; state=False
for i in range(80,len(P)-10):
 hist=V.iloc[max(0,i-120):i].dropna(); qhi=hist.quantile(.70); qlo=hist.quantile(.45); vv=V.iloc[i]
 if not state and len(hist)>40 and vv>qhi: state=True
 elif state and len(hist)>40 and vv<qlo: state=False
 tr=P.iloc[i]/P.iloc[i-20]-1; vol=R.iloc[i-59:i+1].std()*np.sqrt(60); z=tr/(vol+1e-12)
 # In stress, favor defensive relative strength while retaining a small trend component.
 defensive=pd.Series(0.,index=P.columns); defensive[['XAU','US10Y','CN10Y']]=1.
 f=(0.35*(-z)+0.65*defensive.rank(pct=True)) if state else z
 f=f-f.median(); y=P.iloc[i+10]/P.iloc[i]-1
 q=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.f.nunique()>1: rows.append((P.index[i],len(q),q.f.corr(q.y,method='spearman')))
 for s,a in f.items(): signals.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(a),'stress':int(state)})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('candidate=hysteresis_defensive_trend','assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-10-02')]:
 q=x[(x.date>=a)&(x.date<=b)]; print('regime',a,b,'dates',len(q),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6) if len(q)>1 else np.nan)
S=pd.DataFrame(signals).pivot(index='date',columns='symbol',values='signal'); print('turnover',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,20]:
 vals=[]; state=False
 for i in range(80,len(P)-h):
  hist=V.iloc[max(0,i-120):i].dropna(); vv=V.iloc[i]
  if not state and len(hist)>40 and vv>hist.quantile(.70): state=True
  elif state and len(hist)>40 and vv<hist.quantile(.45): state=False
  tr=P.iloc[i]/P.iloc[i-20]-1; vol=R.iloc[i-59:i+1].std()*np.sqrt(60); z=tr/(vol+1e-12); d=pd.Series(0.,index=P.columns); d[['XAU','US10Y','CN10Y']]=1.; f=.35*(-z)+.65*d.rank(pct=True) if state else z
  y=P.iloc[i+h]/P.iloc[i]-1; q=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(q.f.corr(q.y,method='spearman'))
 print('decay',h,'dates',len(vals),'IC',round(np.nanmean(vals),6),'ICIR',round(np.nanmean(vals)/np.nanstd(vals,ddof=1),6))
pd.DataFrame(signals).to_csv('scripts/miner_3_20301031_hysteresis_defensive_trend_signal.csv',index=False)
