import json
from pathlib import Path
import pandas as pd

# current sim date
print("date.json:", Path("../persistent/date.json").read_text())

# trading calendar around the boundary
try:
    from alphacrafter.sim.utils import get_index_daily_data
    df = get_index_daily_data("SPX", days=40)
    if df is not None:
        dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
        print("SPX last 25 dates:", dates[-25:])
except Exception as e:
    print("err", e)
