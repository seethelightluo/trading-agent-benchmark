import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

for sym, fn in [("SPX", get_stock_daily_data), ("VIX", get_index_daily_data), ("XAU", get_stock_daily_data)]:
    try:
        df = fn(sym, days=130)
        print(sym, df.shape, list(df.columns))
        print("dtypes:", df.dtypes.to_dict())
        print("tail:")
        print(df.tail(3))
        s = pd.Series(df["close"].astype(float), index=pd.to_datetime(df["date"]))
        print("series tail:", s.tail(3).tolist())
    except Exception as e:
        print(sym, "ERR:", repr(e))
