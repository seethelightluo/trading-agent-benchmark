
# Persist reproducible signal artifact for admission audit (signal date, no future inputs)
f.loc[dates].to_csv('scripts/miner_2_20321101_vix_amplified_smoothed_residual_signal.csv',index_label='date')
