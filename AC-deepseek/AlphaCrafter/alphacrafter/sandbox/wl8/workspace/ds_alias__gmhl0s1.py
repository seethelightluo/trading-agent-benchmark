import json, numpy as np, pandas as pd

# Load index CSVs beyond api window
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    dcol = df.columns[0]
    if 'date' not in dcol.lower():
        df = df.rename(columns={dcol:'date'})
    df['date'] = pd.to_datetime(df['date'].astype(str).str[:10])
    return df

# Load asset data from ../persistent/stock_data if exists
import os
sd = '../persistent/stock_data'
print('stock_data files:', len(os.listdir(sd)) if os.path.isdir(sd) else 'none')
print(os.listdir(sd)[:20] if os.path.isdir(sd) else '')