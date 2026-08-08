"""Scheduled revalidation: persisted inverse residual VIX--EURUSD joint-stress loading expansion only."""
import json, numpy as np, pandas as pd
src=open('scripts/miner_2_20310501_residual_vix_usdcny_joint_stress_loading_expansion_20_60d.py',encoding='utf8').read()
prefix=src.split('# A continuous joint-risk driver:')[0].replace("END=pd.Timestamp('2031-04-30')","END=pd.Timestamp('2031-08-06')")
exec(prefix,globals())
def obsret(name):
    x=pd.read_csv('../persistent/index_data/'+name+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END,'close'].astype(float).pct_change().reindex(p.index)
    return x/(x.rolling(60,min_periods=40).std()+1e-12)
# Exact persisted definition: inverse of recent-minus-structural residual loading to joint VIX-up/EURUSD-down shocks.
driver=np.maximum(obsret('VIX'),0)*np.maximum(-obsret('EURUSD'),0)
f=-(beta(e,driver,20,14)-beta(e,driver,60,42))
print('FACTOR inverse_residual_vix_eurusd_joint_stress_loading_expansion_20_60d REVALIDATION','validation_end',END.date(),'panel',p.index.min().date(),p.index.max().date(),'universe',len(A),'library',len(lib),'driver_nonnull',round(driver.notna().mean(),6),'driver_nonzero',round((driver>0).mean(),6))
metrics={}; ics={}
for h in [1,5,10,20]:
    fw=p.shift(-h)/p-1; out=[]; ns=[]
    for t in f.index:
        z=pd.DataFrame({'f':f.loc[t],'y':fw.loc[t]}).dropna()
        if len(z)>=8 and z.f.nunique()>1:
            q=z.f.corr(z.y,method='spearman')
            if pd.notna(q): out.append((t,q)); ns.append(len(z))
    z=pd.Series(dict(out),dtype=float); ics[h]=z; sd=z.std(ddof=1)
    metrics[h]={'daily_paper_ic':z.mean(),'daily_paper_icir':z.mean()/sd,'ic_std':sd,'ic_standard_error':sd/np.sqrt(len(z)),'ic_hit_ratio':(z>0).mean(),'ic_dates':len(z),'mean_valid_instruments':np.mean(ns)}
    print('HORIZON',h,json.dumps({k:round(float(v),6) for k,v in metrics[h].items()}))
for name,mask in [('2020_24',ics[10].index<pd.Timestamp('2025-01-01')),('2025_26',(ics[10].index>=pd.Timestamp('2025-01-01'))&(ics[10].index<pd.Timestamp('2027-01-01'))),('2027_onward',ics[10].index>=pd.Timestamp('2027-01-01'))]:
    z=ics[10][mask]; print('REGIME10',name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
    z=rk.iloc[[i-1,i]].T.dropna()
    if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('COVERAGE',round(f.notna().mean().mean(),6),'VALID_CELLS',int(f.notna().sum().sum()),'RANK_TURNOVER',round(float(np.nanmean(turn)),6),'TURNOVER_DATES',len(turn))
screen=[]
for n,s in sorted(lib.items()):
    z=pd.concat([f.stack().rename('f'),s.stack().rename('s')],axis=1).dropna(); rho=z.f.corr(z.s,method='spearman')
    if pd.notna(rho): screen.append((abs(rho),n,rho,len(z)))
mx,n,rho,c=max(screen); print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',n,'rho',round(rho,6),'cells',c)
print('DECAY',json.dumps({str(h):{'ic':round(q['daily_paper_ic'],6),'icir':round(q['daily_paper_icir'],6),'dates':q['ic_dates']}for h,q in metrics.items()}))
