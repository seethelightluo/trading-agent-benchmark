import pandas as pd
p='scripts/miner_2_20281005_volcompression_signal.csv'
d=pd.read_csv(p); d['signal']=-d['signal']; d.to_csv('scripts/miner_2_20281005_volcompression_signal_oriented.csv',index=False)
print(len(d), d['signal'].notna().mean())
