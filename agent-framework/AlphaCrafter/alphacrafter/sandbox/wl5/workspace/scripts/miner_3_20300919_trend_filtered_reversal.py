import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4000); d.date=pd.to_datetime(d.date); C[s]=d.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
P=pd.DataFrame(C).sort_index(); R=np.log(P).diff(); rows=[]; sigs=[]
for i in range(61,len(P)-10):
 short=P.iloc[i]/P.iloc[i-5]-1; long=P.iloc[i]/P.iloc[i-60]-1; v=R.iloc[i-19:i+1].std()
 # buy short-term losers only when long-term trend is positive; cross-sectional comparable and risk scaled
 gate=(long>long.median()).astype(float)
 f=(-short/(v+1e-12))*(0.25+0.75*gate)
 y=P.iloc[i+10]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((P.index[i],len(z),z.f.corr(z.y,method='spearman')))
 for s,vv in f.items(): sigs.append({'date':str(P.index[i].date()),'symbol':s,'signal':float(vv) if pd.notna(vv) else np.nan})
x=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); m=x.ic.mean(); sd=x.ic.std(ddof=1)
print('assets',15,'dates',len(x),'meanN',round(x.n.mean(),2),'coverage',round(x.n.sum()/(len(x)*15),5),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((x.ic>0).mean(),5))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2027-12-31'),('2028-01-01','2029-12-31'),('2030-01-01','2030-09-18')]:
 z=x[(x.date>=a)&(x.date<=b)]; print(a,b,'dates',len(z),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6) if len(z)>1 else np.nan)
S=pd.DataFrame(sigs).pivot(index='date',columns='symbol',values='signal'); print('rank_turnover_proxy',round(S.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
for h in [5,10,20]:
 rr=[]
 for i in range(61,len(P)-h):
  short=P.iloc[i]/P.iloc[i-5]-1; long=P.iloc[i]/P.iloc[i-60]-1; v=R.iloc[i-19:i+1].std(); f=(-short/(v+1e-12))*(0.25+0.75*(long>long.median())); y=P.iloc[i+h]/P.iloc[i]-1; z=pd.DataFrame({'f':f,'y':y}).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1: rr.append(z.f.corr(z.y,method='spearman'))
 print('decay',h,'dates',len(rr),'IC',round(np.nanmean(rr),6),'ICIR',round(np.nanmean(rr)/np.nanstd(rr,ddof=1),6))
pd.DataFrame(sigs).to_csv('scripts/miner_3_20300919_trend_filtered_reversal_signal.csv',index=False)
