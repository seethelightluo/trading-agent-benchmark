"""Validate one idea: 20-observation downside-asymmetry resilience.
Signal is negative downside semi-volatility divided by total realized volatility;
higher is a return path with less downside variability.  Cross-sectional daily
Spearman ICs use only dates with >=8 names and forward returns are daily
# [line 5 missing]
# [line 6 missing]
# [line 7 missing]
# [line 8 missing]
# [line 9 missing]
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
# [line 11 missing]
assets=get_account_dict()['watch_list']
# [line 13 missing]
# [line 14 missing]
    x=get_stock_daily_data(a,5000).copy(); x['date']=pd.to_datetime(x['date'])
# [line 16 missing]
# [line 17 missing]
# [line 18 missing]
# [line 19 missing]
# [line 20 missing]
# [line 21 missing]
# library definitions reconstructed from persisted expressions
# [line 23 missing]
# [line 24 missing]
# [line 25 missing]
# [line 26 missing]
# [line 27 missing]
# [line 28 missing]
# [line 29 missing]
# [line 30 missing]
# [line 31 missing]
# [line 32 missing]
 x=get_stock_daily_data(a,5000).copy(); x['date']=pd.to_datetime(x['date']); v[a]=pd.to_numeric(x.set_index('date').get('volume'),errors='coerce')
# [line 34 missing]
# [line 35 missing]
# [line 36 missing]
# [line 37 missing]
# [line 38 missing]
# [line 39 missing]
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); counts.append(len(z))
# [line 41 missing]
# [line 42 missing]
# [line 43 missing]
# [line 44 missing]
# [line 45 missing]
# [line 46 missing]
# [line 47 missing]
# [line 48 missing]
 if len(z)>=8: ic.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
# [line 50 missing]
# [line 51 missing]
# [line 52 missing]
# maximum absolute Spearman signal-cell correlation vs each admitted factor
# [line 54 missing]
# [line 55 missing]
 print('LIBCORR',name,round(z.a.corr(z.b,method='spearman'),6),'cells',len(z))