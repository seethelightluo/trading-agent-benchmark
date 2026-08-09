import runpy
ns=runpy.run_path('scripts/miner_2_20301128_revalidate_stress_duration_weighted_peer_resilience_reversal_60.py')
P,F,out=ns['P'],ns['F'],ns['out']
print('PRICE_NON_NULL_ENDPOINTS',P.notna().apply(lambda s:s.index[s].max().date()).value_counts().to_dict())
print('FACTOR_LAST_VALID',F.dropna(how='all').index.max().date())
for h,q in out.items(): print('IC_RANGE',h,q.date.min().date(),q.date.max().date())
