# append artifacts for all tested variants
for _kind in ['riskoff_reversal','breadth']:
    _q=make(_kind,1)
    _q[['date','asset','f']].to_csv('scripts/miner_2_20270128_'+_kind+'_signal.csv',index=False)
