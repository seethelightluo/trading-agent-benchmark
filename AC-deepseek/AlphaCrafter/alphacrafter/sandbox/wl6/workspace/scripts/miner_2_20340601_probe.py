"""miner_2 2034-06-01 probe: establish visible date and panel state."""
import pandas as pd
TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
            'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
for ddir, name, syms in [("../persistent/stock_data","stock",TRADABLE),
                          ("../persistent/index_data","index",OBS)]:
    print("==", name, "==", flush=True)
    for s in syms:
        try:
            df = pd.read_csv(f"{ddir}/{s}.csv", parse_dates=["date"])
            print(s, len(df), df["date"].min().date(), "->", df["date"].max().date(), flush=True)
        except Exception as e:
            print(s, "ERR", e, flush=True)