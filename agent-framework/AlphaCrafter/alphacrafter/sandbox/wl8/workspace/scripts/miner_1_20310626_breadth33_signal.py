import pandas as pd
# Reuse the validated breadth-threshold implementation and emit the selected signal artifact.
exec(open('scripts/miner_1_20310626_breadth_thresholds.py').read().split("for th in [.33,.40,.50]:")[0])
th=.33; f=mom.where(br>=th,-mom); f.to_csv('scripts/miner_1_20310626_breadth33_momentum_signal.csv')
