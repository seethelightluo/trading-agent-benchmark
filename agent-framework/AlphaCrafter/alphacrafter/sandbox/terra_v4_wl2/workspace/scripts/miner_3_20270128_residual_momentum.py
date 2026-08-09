import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in assets}
r={a:p[a].pct_change() for a in assets}
# idiosyncratic medium momentum: asset 20d return minus contemporaneous cross-sectional median 20d return
raw={a:p[a]/p[a].shift(20)-1 for a in assets}
allidx=sorted(set().union(*[set(x.index) for x in p.values()]))
rows=[]; sigrows=[]
for dt in allidx:
    vals={a:raw[a].get(dt,np.nan) for a in assets}
    med=np.nanmedian(list(vals.values())) if np.isfinite(list(vals.values())).sum()>=8 else np.nan
    for a in assets:
        f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
        sigrows.append((dt,a,f))
    for h in [1,5,10]:
        fwd=[]; fac=[]
        for a in assets:
            if dt not in p[a].index: continue
            ix=p[a].index.get_loc(dt); f=vals[a]-med if np.isfinite(vals[a]) and np.isfinite(med) else np.nan
            if ix+h>=len(p[a]) or not np.isfinite(f): continue
            y=p[a].iloc[ix+h]/p[a].iloc[ix]-1
            if np.isfinite(y): fac.append(f);fwd.append(y)
        if len(fac)>=8:
            rows.append((dt,h,spearmanr(fac,fwd).statistic,len(fac)))
df=pd.DataFrame(rows,columns=['date','h','ic','n']);
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.mean()/15,'IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(),'hit',(x.ic>0).mean())
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),5),round(z.mean()/z.std(),5) if len(z)>1 else None)
# daily artifact
pd.DataFrame(sigrows,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_3_20270128_residual_momentum.csv',index=False)
# turnover rank changes
wide=pd.DataFrame(sigrows,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal')
ranks=wide.rank(axis=1,pct=True); print('rank turnover',ranks.diff().abs().mean(axis=1).mean())
