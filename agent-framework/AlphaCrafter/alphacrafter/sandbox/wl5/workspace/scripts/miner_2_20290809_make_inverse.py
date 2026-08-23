import pandas as pd
p='scripts/miner_2_20290809_relative_strength_20d_signal.csv'
d=pd.read_csv(p); d['signal']=-d['signal']; d.to_csv('scripts/miner_2_20290809_relative_strength_reversal_20d_signal.csv',index=False)
