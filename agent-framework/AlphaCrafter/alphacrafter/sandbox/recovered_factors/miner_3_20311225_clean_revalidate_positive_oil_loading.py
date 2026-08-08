"""Clean-calendar revalidation of one admitted factor: positive WTI shock loading contraction."""
import json, numpy as np, pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-12-24')
def series(a):
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float)
    return x
raw={a:series(a) for a in A}
cal=pd.date_range(max(x.index.min() for x in raw.values()),END,freq='B')
p=pd.DataFrame(raw).reindex(cal).ffill()
r=p.pct_change(); market=r.mean(axis=1)
def beta(x,y,w,n):
    return pd.DataFrame({a:x[a].rolling(w,min_periods=n).cov(y)/(y.rolling(w,min_periods=n).var()+1e-12) for a in A})
bm=beta(r,market,60,40); e=r-bm.mul(market,axis=0)
oil=r['WTI'].clip(lower=0)
f=beta(e,oil,60,42)-beta(e,oil,20,14)
print('FACTOR revalidation_residual_positive_oil_change_shock_loading_contraction_20_60d')
print('VALIDATION_END',END.date(),'COMMON_CALENDAR_DATES',len(cal),'UNIVERSE',len(A),'RAW_COVERAGE',round(float(f.notna().mean().mean()),6),'VALID_CELLS',int(f.notna().sum().sum()))
metrics={}; ics={}
for h in [1,5,10,20]:
    fw=p.shift(-h)/p-1; out=[]; ns=[]
    for t in f.index[:-h]:
        z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
        if len(z)>=8 and z.f.nunique()>1:
            q=z.f.corr(z.y,method='spearman')
            if pd.notna(q): out.append((t,q)); ns.append(len(z))
    x=pd.Series(dict(out)); ics[h]=x; sd=x.std(ddof=1)
    d={'daily_paper_ic':x.mean(),'daily_paper_icir':x.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(x)),'ic_hit_ratio':(x>0).mean(),'ic_dates':len(x),'mean_valid_instruments':np.mean(ns)}
    metrics[h]=d; print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in d.items()}))
for name,mask in [('2020_24',ics[10].index<'2025-01-01'),('2025_26',(ics[10].index>='2025-01-01')&(ics[10].index<'2027-01-01')),('2027_onward',ics[10].index>='2027-01-01')]:
 x=ics[10][mask]; print('REGIME10',name,'DATES',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'HIT',round((x>0).mean(),6))
rk=f.rank(axis=1,pct=True); tos=[]
for i in range(1,len(rk)):
 z=rk.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('RANK_TURNOVER',round(float(np.nanmean(tos)),6),'TURNOVER_DATES',len(tos))
print('DECAY',json.dumps({str(h):{'ic':round(float(d['daily_paper_ic']),6),'icir':round(float(d['daily_paper_icir']),6),'dates':d['ic_dates']} for h,d in metrics.items()}))
print('LIBRARY_SCREEN_STATUS NOT_COMPUTED: authoritative reconstruction of all 30 currently admitted signals is unavailable; no admission/revalidation orthogonality claim.')
