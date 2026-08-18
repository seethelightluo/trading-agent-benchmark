"""Verify live factor recomputation against persisted signal artifacts (truncated to signal end 2026-07-29)."""
import json, numpy as np, pandas as pd

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
END = "2026-07-29"

def load_close(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= END].set_index("date")["close"].astype(float)
    return df

closes = {a: load_close(a) for a in ASSETS}
rets = pd.concat([closes[a].pct_change().rename(a) for a in ASSETS], axis=1).dropna()
market = rets.mean(axis=1)

# --- factor 1: rel_mom_20d_skip5: mom = close.shift(5)/close.shift(25)-1, demeaned ---
mom = pd.DataFrame({a: closes[a].shift(5) / closes[a].shift(25) - 1 for a in ASSETS})
rel_mom = mom.sub(mom.median(axis=1), axis=0)

# --- factor 2: beta_ew_60d ---
def rolling_beta(r, m, w=60):
    out = {}
    for a in r.columns:
        s = pd.concat([r[a].rename("y"), m.rename("x")], axis=1).dropna()
        cov = s["y"].rolling(w).cov(s["x"])
        var = s["x"].rolling(w).var()
        out[a] = (cov / var)
    return pd.DataFrame(out, index=r.index)

beta_ew = rolling_beta(rets, market, 60)

# --- factor 3: downside_vol_ratio_20 = -(downside semi-vol / total vol) ---
def downside_ratio(r, w=20):
    out = {}
    for a in r.columns:
        s = r[a]
        neg = s.clip(upper=0)
        semi = (neg ** 2).rolling(w).mean() ** .5
        tot = s.rolling(w).std()
        out[a] = -(semi / tot)
    return pd.DataFrame(out, index=r.index)

downside = downside_ratio(rets, 20)

# --- factor 4: max_ret_20d ---
max_ret = rets.rolling(20).max()

# --- factor 5: eurusd_beta_cond_60x20 ---
eur = pd.read_csv("../persistent/index_data/EURUSD.csv")
eur["date"] = pd.to_datetime(eur["date"])
eur = eur[eur["date"] <= END].set_index("date")["close"].astype(float)
eur_ret = eur.pct_change()
eur_20 = eur / eur.shift(20) - 1
eur_beta = rolling_beta(rets, eur_ret, 60)
eur_cond = eur_beta.mul(eur_20, axis=0)

# --- factor 6: corr_ew_60 (mean pairwise rolling corr) ---
def corr_ew(r, w=60):
    out = {}
    for a in r.columns:
        others = [b for b in r.columns if b != a]
        cs = [r[a].rolling(w).corr(r[b]) for b in others]
        out[a] = sum(cs) / len(cs)
    return pd.DataFrame(out, index=r.index)

corr_ew = corr_ew(rets, 60)

# --- factor 7: kurt_20d_skip5: rolling 20d kurtosis of returns, shifted by 5 ---
kurt = rets.rolling(20).kurt().shift(5)

factors = {
    "rel_mom_20d_skip5": rel_mom,
    "beta_ew_60d": beta_ew,
    "downside_vol_ratio_20": downside,
    "max_ret_20d": max_ret,
    "eurusd_beta_cond_60x20": eur_cond,
    "corr_ew_60": corr_ew,
    "kurt_20d_skip5": kurt,
}

last = rets.index[-1]
print("last date:", last.date())
for fid, fdf in factors.items():
    np_sig = np.load(f"factors/{fid}.signal.npy") if __import__("os").path.exists(f"factors/{fid}.signal.npy") else None
    live = fdf.loc[last]
    if np_sig is None:
        print(fid, "no npy; live:", live.round(3).to_dict())
        continue
    # align npy rows to dates: reconstruct date index from provenance
    prov = json.load(open(f"factors/{fid}.json"))["artifact_provenance"]
    d0 = pd.to_datetime(prov["dates_first"]); d1 = pd.to_datetime(prov["dates_last"])
    # dates assumed business days; use rets index aligned by position
    n = np_sig.shape[0]
    sig = pd.DataFrame(np_sig, columns=ASSETS, index=rets.index[-n:])
    sig = sig.loc[last]
    both = pd.concat([live.rename("live"), sig.rename("npy")], axis=1).dropna()
    rank_corr = both["live"].rank().corr(both["npy"].rank())
    val_corr = both["live"].corr(both["npy"])
    print(f"{fid}: rank_corr={rank_corr:.3f} val_corr={val_corr:.3f} n={len(both)}")
    if len(both):
        print("   live:", {a: round(float(both.loc[a,'live']),4) for a in both.index[:15]})
        print("   npy :", {a: round(float(both.loc[a,'npy']),4) for a in both.index[:15]})
