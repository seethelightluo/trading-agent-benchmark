import pandas as pd
p='scripts/miner_2_20320628_persistent_trend_signal.csv'
d=pd.read_csv(p); d['signal']=-d['signal']; d.to_csv('scripts/miner_2_20320628_persistent_trend_reversal_signal.csv',index=False)
