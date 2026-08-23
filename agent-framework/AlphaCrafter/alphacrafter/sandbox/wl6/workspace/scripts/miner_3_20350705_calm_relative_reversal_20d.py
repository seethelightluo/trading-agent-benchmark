import pandas as pd
p='scripts/miner_3_20350705_calm_relative_momentum_signal.csv'
f=pd.read_csv(p,index_col='date',parse_dates=True)
# Invert the validated anti-predictive calm-momentum ranking: high values favor future reversal.
(-f).to_csv('scripts/miner_3_20350705_calm_relative_reversal_signal.csv',index_label='date')
print('wrote inverted signal artifact',(-f).shape,'dates',f.index.min(),f.index.max(),'assets',f.shape[1])
