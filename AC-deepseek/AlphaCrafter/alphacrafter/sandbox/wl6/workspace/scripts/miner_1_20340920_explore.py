import json, os
import pandas as pd

with open('../persistent/date.json') as f:
    dj = json.load(f)
print("current_date:", dj.get('current_date'), "visible_through:", dj.get('visible_through'))
print("n trading days:", len(dj.get('trading_days', [])))

for base, key in [('../persistent/stock_data', 'stock'), ('../persistent/index_data', 'index')]:
    print("\n=== ", base, "===")
    if os.path.isdir(base):
        files = sorted(os.listdir(base))
        print("n files:", len(files), files)
        if files:
            p = os.path.join(base, files[0])
            df = pd.read_csv(p)
            print("sample file:", os.path.basename(p))
            print("cols:", list(df.columns))
            print(df.tail(3).to_string())
    else:
        print("MISSING")