# [line 1 missing]
# [line 2 missing]
# [line 3 missing]
# [line 4 missing]
# [line 5 missing]
# [line 6 missing]
# [line 7 missing]
# [line 8 missing]
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
# [line 10 missing]
# [line 11 missing]
# [line 12 missing]
 # admitted library signal definitions, reconstructed from published definitions
# [line 14 missing]
# [line 15 missing]
# [line 16 missing]
# [line 17 missing]
# [line 18 missing]
# [line 19 missing]
# [line 20 missing]
# [line 21 missing]
# [line 22 missing]
# [line 23 missing]
   out.append((dt,z.factor.corr(z.forward,method='spearman'))); covers.append(len(z)/15)
# [line 25 missing]
# [line 26 missing]
# [line 27 missing]
 print(f'h={h} dates={len(ic)} meanIC={ic.mean():.6f} ICIR={ir:.6f} hit={(ic>0).mean():.4f} IC_se={ic.std(ddof=1)/np.sqrt(len(ic)):.6f} coverage={cov:.4f}')
# [line 29 missing]
  x=ic[mask]; print(f'  {label}: n={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f}')
# [line 31 missing]
# [line 32 missing]
# [line 33 missing]
# [line 34 missing]
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
# [line 36 missing]
pairs=pd.concat([f.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna()
rho=pairs.candidate.corr(pairs.library,method='spearman')
# [line 39 missing]
print(f'LIBRARY: records={len(glob.glob(