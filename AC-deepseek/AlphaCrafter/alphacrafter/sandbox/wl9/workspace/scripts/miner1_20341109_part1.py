def roll_ac(ser, w=120):
    def _ac(x):
        xc = x[~np.isnan(x)]
        if len(xc) < 5 or np.std(xc) < 1e-12: return np.nan
        return np.corrcoef(xc[:-1], xc[1:])[0,1]
    return ser.rolling(w).apply(_ac, raw=True)

# ============================
# PART 1: REVALIDATE EXISTING EFFECTIVE FACTORS (2yr window)
# ============================
print("\n=== PART 1: REVALIDATE EXISTING EFFECTIVE FACTORS (recent 500d) ===", flush=True)
mask2 = close.index >= (VISIBLE - pd.Timedelta(days=760))
close2 = close[mask2]
rets2 = close2.pct_change().dropna()
fwd2_10d = rets2.shift(-10).rolling(10).sum()
vix2 = vix[vix.index >= VISIBLE - pd.Timedelta(days=800)] if vix is not None else None
dVIX2 = vix2.pct_change() if vix2 is not None else None
dCNY2 = cny[cny.index >= VISIBLE - pd.Timedelta(days=800)].pct_change() if cny is not None else None
dDXY2 = dxy[dxy.index >= VISIBLE - pd.Timedelta(days=800)].pct_change() if dxy is not None else None

lib = {}
lib['beta_VIX_60'] = -beta_win(rets2, dVIX2, 60)          # direction -1 factor value stored as -beta
lib['kaufman_eff_20d'] = kaufman(close_2, 20)
lib['mom_120d_skip5'] = close_2.pct_change(120)
lib['mom_10d_skip5'] = close_2.pct_change(10)
lib['bb_width_20d'] = (close_2.rolling(20).max() - close_2.rolling(20).min())/close_2.rolling(20).mean()
lib['vol_z_20d'] = (rets2.rolling(20).std() - rets2.rolling(60).std())/rets2.rolling(60).std()
lib['cny_beta_60'] = beta_win(rets2, dCNY2, 60)
lib['dxy_corr_change_20_60'] = corr_win(rets2, dDXY2, 20) - corr_win(rets2, dDXY2, 60)
lib['ac1_120d'] = -close_2.apply(lambda c: roll_ac(c, 120))   # negated (low ac1 favored)
lib['skew_20d'] = rets2.rolling(20).skew()
lib['rng_pos_20d'] = round(((high_df[high_df.index.isin(close_2.index)] - low_df[low_df.index.isin(close_2.index)]).replace(0,np.nan)).div(close_2), 4)
lib['streak_len_14'] = close_2.pct_change().apply(lambda v: v.gt(0).astype(int))
from scipy import stats as _st
def rolling_skew(ser, w=20):
    return ser.rolling(w).apply(lambda x: _st.skew(x) if len(x)>=w else np.nan, raw=True)

for nm, f in lib.items():
    if f is None: 
        print(f"  [--] {nm:26s} None", flush=True); continue
    _ = report('LIB:'+nm, f)
print("(LIB flag reflects recent ~500d window, for aging info only)", flush=True)