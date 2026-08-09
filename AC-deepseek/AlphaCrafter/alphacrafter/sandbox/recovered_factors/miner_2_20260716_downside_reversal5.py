# [line 1 missing]
# [line 2 missing]
# [line 3 missing]
# [line 4 missing]
from scipy.stats import spearmanr
# [line 6 missing]
# [line 7 missing]
# [line 8 missing]
# [line 9 missing]
    p=(pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date'])
       .query('date<=@END').drop_duplicates('date').set_index('date').close.astype(float).sort_index())
# [line 12 missing]
# [line 13 missing]
    downside=r.clip(upper=0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
    factors[a]=(-p.pct_change(5,fill_method=None)/downside.replace(0,np.nan))
# [line 16 missing]
# [line 17 missing]
# [line 18 missing]
# [line 19 missing]
# [line 20 missing]
# [line 21 missing]
# [line 22 missing]
# [line 23 missing]
# [line 24 missing]
            out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)); n.append(len(z))
# [line 26 missing]
# [line 27 missing]
# [line 28 missing]
# [line 29 missing]
# [line 30 missing]
        if len(z)>=8: changes.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    return ic, {'daily_paper_ic':float(m),'daily_paper_icir':float(m/sd),'ic_std':float(sd),'ic_hit_ratio':float((ic>0).mean()),'n_dates':len(ic),'mean_valid_instruments':float(np.mean(n)),'turnover':float(np.mean(changes)),'yearly_ic':{str(y):float(x.mean()) for y,x in ic.groupby(ic.index.year)}}
# [line 33 missing]
# [line 34 missing]
# [line 35 missing]
# [line 36 missing]
# [line 37 missing]
# [line 38 missing]
# [line 39 missing]
# [line 40 missing]
# [line 41 missing]
# [line 42 missing]
# Correlate complete aligned panel values (Spearman); evidence for every existing admitted signal is mandatory.
# [line 44 missing]
# [line 45 missing]
    lib=pd.read_pickle(path).reindex(index=factor.index,columns=ASSETS)
# [line 47 missing]
    rho=spearmanr(z.new,z.lib).statistic if len(z)>=8 else np.nan
    print('LIBRARY_CORR',os.path.basename(path),'n_pairs',len(z),'spearman',rho)
# [line 50 missing]
print('FACTOR downside_volatility_adjusted_reversal_5d')
# [line 52 missing]
print('DECAY',json.dumps({str(h):{'ic':allmet[h]['daily_paper_ic'],'icir':allmet[h]['daily_paper_icir'],'n_dates':allmet[h]['n_dates']} for h in allmet}))
# [line 54 missing]
factor.to_pickle('scripts/miner_2_20260716_downside_reversal5_signal.pkl')