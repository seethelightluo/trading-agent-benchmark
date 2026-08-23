import numpy as np, pandas as pd

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Drawdown recovery: lagged distance above prior 60d trough, scaled by prior 20d vol.
# Use only through t-1; forward return t+1..t+5.
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']) for s in U}
rows=[]
for s,df in D.items():
    if df is None or len(df)<300: continue
    x=df.copy(); x['date']=pd.to_datetime(x['date']); x=x.loc[x['date']<='2029-08-22']; x=x.sort_values('date').set_index('date')
    p=x['close'].astype(float); r=p.pct_change()
    trough=p.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std()
    # lag everything, factor is recovery from trough: higher = recovered/strong; use bounded log ratio
    f=(np.log(p/trough)/vol.replace(0,np.nan)).shift(1)
    fr=np.log(p.shift(-5)/p.shift(-1))
    z=pd.DataFrame({'symbol':s,'factor':f,'fwd':fr})
    z=z.reset_index().rename(columns={'index':'date'}); rows.append(z)
a=pd.concat(rows,ignore_index=True).dropna()
# dates with >=8 names
ics=[]; ranks=[]
for dt,g in a.groupby('date'):
    if len(g)>=8 and g['factor'].nunique()>1 and g['fwd'].nunique()>1:
        ics.append((dt,g['factor'].corr(g['fwd'],method='spearman')))
        ranks.append((dt,g[['symbol','factor']].sort_values('factor')['symbol'].tolist()))
ics=pd.Series(dict(ics)).dropna(); ics.index=pd.to_datetime(ics.index)
# turnover: average rank ordering changes per asset across adjacent observations (approx signal sign/rank changes)
piv=a.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
to=piv.diff().abs().mean().mean()
print('dates',len(ics),'avg instruments',round(a.groupby('date').size().loc[a.groupby('date').size()>=8].mean(),2),'coverage',round(len(a)/(len(a['date'].unique())*15),4))
for h in [1,5,10,20]:
    rr=[]
    for s,df in D.items():
        if df is None: continue
        p=df.loc[df['date']<='2029-08-22'].sort_values('date').set_index('date')['close'].astype(float); r=p.pct_change(); trough=p.rolling(60,min_periods=40).min(); vol=r.rolling(20,min_periods=15).std(); f=(np.log(p/trough)/vol.replace(0,np.nan)).shift(1); fr=np.log(p.shift(-h)/p.shift(-1)); rr.append(pd.DataFrame({'date':p.index,'factor':f,'fwd':fr}))
    q=pd.concat([z.dropna() for z in rr]).reset_index(drop=True); out=[]
    for dt,g in q.groupby('date'):
        if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1: out.append(g.factor.corr(g.fwd,method='spearman'))
    ss=pd.Series(out).dropna(); print('horizon',h,'IC',round(ss.mean(),6),'ICIR',round(ss.mean()/ss.std(),6),'hit',round((ss>0).mean(),4),'n',len(ss))
print('turnover',round(float(to),6))
# regimes for 5d from original dates
for name,mask in [('2026',ics.index.year==2026),('2027-28',ics.index.year.isin([2027,2028])),('recent360',ics.index>=ics.index.max()-pd.Timedelta(days=360)),('recent180',ics.index>=ics.index.max()-pd.Timedelta(days=180))]:
 ss=ics[mask]; print(name,'n',len(ss),'IC',round(ss.mean(),6),'ICIR',round(ss.mean()/ss.std(),6) if ss.std()>0 else None)
