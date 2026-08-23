import pandas as pd
from pathlib import Path
base = Path('../persistent')
for f in ['index_data/VIX.csv','stock_data/SPX.csv','stock_data/000300.SH.csv']:
    df = pd.read_csv(base/f, parse_dates=True)
    print(f, df.shape)
    print(df.columns.tolist())
    print(df.head(3))
    print(df.tail(3))
    print()