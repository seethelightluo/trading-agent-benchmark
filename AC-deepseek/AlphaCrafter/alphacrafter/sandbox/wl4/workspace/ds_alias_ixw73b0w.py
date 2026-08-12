import pandas as pd, glob, os
files = sorted(glob.glob('../persistent/stock_data/*.csv'))
for f in files[:3]:
    df = pd.read_csv(f)
    print(os.path.basename(f), "cols:", list(df.columns), "rows:", len(df), "last date:", df['date'].iloc[-1] if 'date' in df.columns else '?')
# check all last dates
lasts = {}
for f in files:
    df = pd.read_csv(f)
    lasts[os.path.basename(f)] = str(df['date'].iloc[-1])
print(sorted(set(lasts.values())))
