"""miner_2 2034-05-15: quick probe of data availability and current date."""
import sys, warnings
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from factor_research_lib import load_panels, close_panel

panels = load_panels(days=6000)
closes = close_panel(panels)
print("n_tradable_panels:", len([a for a in panels if a in ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]]))
print("n_macro_panels:", len([a for a in panels if a in ["VIX","DXY","USDCNY","USDJPY","EURUSD"]]))
print("close panel shape:", closes.shape)
print("date range:", closes.index.min(), "->", closes.index.max())
print("last 5 dates:", list(closes.index[-5:].strftime("%Y-%m-%d")))
print("n_valid per asset (last 250d):")
print(closes.tail(250).notna().sum())
