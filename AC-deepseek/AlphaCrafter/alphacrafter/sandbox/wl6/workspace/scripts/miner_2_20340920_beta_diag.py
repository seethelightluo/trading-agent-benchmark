import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
VISIBLE="2034-09-19"
TRADABLE=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(sym,ddir):
    df=pd.read_csv(f"{ddir}/{sym}.csv",parse_dates=["date"])
    return df[df["date"]<=pd.Timestamp(VISIBLE)].set_index("date")["close"].astype(float).sort_index()
px=pd.DataFrame({s:load(s,"../persistent/stock_data") for s in TRADABLE})
vix=load("VIX","../persistent/index_data/").reindex(px.index)
r=px.pct_change()
print("vix nonnull:", int(vix.notna().sum()), "of", len(vix), "range", vix.dropna().index.min().date(), vix.dropna().index.max().date())
vcov=r.rolling(60).cov(vix)
print("vcov nonnull:", int(vcov.notna().sum().sum()), "of", vcov.size)
vvar=vix.rolling(60).var()
print("vvar nonnull:", int(vvar.notna().sum()))
beta=vcov/vvar
print("beta rows status: total NaN rows", int(beta.isna().all(axis=1).sum()))
print("max valid dates:", beta.notna().sum(axis=1).max(), "median", beta.notna().sum(axis=1).median())
print("sample tail beta nonnull per row (last 5):")
print(beta.notna().sum(axis=1).tail(5))
print("sample beta tail:")
print(beta.tail(3).round(3))