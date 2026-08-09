import pandas as pd, numpy as np
f=pd.read_pickle('scripts/miner_1_20301212_common_stress_overnight_gap_resilience_response_60obs_candidate_signal.pkl')
rk=f.rank(axis=1,pct=True); vals=[]
for i in range(1,len(rk)):
    q=pd.DataFrame({'prior':rk.iloc[i-1].to_numpy(),'current':rk.iloc[i].to_numpy()}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(q)>=8:
        c=q['prior'].corr(q['current'],method='spearman')
        if pd.notna(c): vals.append(1-c)
print('turnover=',float(np.mean(vals)) if vals else None,'pairs=',len(vals),'median=',float(np.median(vals)) if vals else None)
print('cells=',int(f.notna().sum().sum()),'total=',f.size)
