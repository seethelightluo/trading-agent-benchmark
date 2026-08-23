"""debug why beta factors return NO IC."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
VISIBLE="2034-09-26"
TRADABLE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym,ddir):
    df=pd.read_csv(f"{ddir}/{sym}.csv",parse_dates=["date"]).drop_duplicates(subset='date',keep='last')
    return df[df["date"]<=pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float).sort_index()
px=pd.DataFrame({s:load(s,"../persistent/stock_data") for s in TRADABLE})
vix=load("VIX","../persistent/index_data/").reindex(px.index)
r=px.pct_change()
vixc=vix.replace(0,np.nan); vixret=vixc.pct_change()
beta_vix = r.rolling(60).cov(vixret)/vixret.rolling(60).var()
print("vixret NaN total:", int(vixret.isna().sum()))
print("beta_vix valid per date - last 10 rows:", beta_vix.notna().sum(axis=1).tail(10).tolist())
# Check VIX values recent
print("VIX last 5:", vix.dropna().tail(5).round(1).tolist())
print("VIX tail dates:", vix.dropna().tail(5).index.strftime('%Y-%m-%d').tolist())
# Try alternative: compute beta manually row by row for a sample
res = beta_vix.rolling(60).sum().sum()  # just to force
np.set_printoptions(suppress=True)
print("sample beta_vix last 3 rows:")
print(beta_vix.tail(3).round(3))
print("beta_vix has any nonnull:", beta_vix.notna().any().any())