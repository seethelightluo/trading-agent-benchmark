# Patch miner_3_20271007_screen_batch.py: new visible date, timestamp, fresh candidates
p = 'scripts/miner_3_20271007_screen_batch.py'
s = open(p).read()

s = s.replace('VISIBLE = "2027-09-22"', 'VISIBLE = "2027-10-06"')
s = s.replace('"last_validated": "2027-09-23"', '"last_validated": "2027-10-07"')
s = s.replace('2027-09-23 cycle (data visible through 2027-09-22)',
              '2027-10-07 cycle (data visible through 2027-10-06)')

insert = '''
# K) fresh candidates for 2027-10-07 cycle
def rsi14(pxx):
    d = pxx.diff()
    up = d.clip(lower=0).rolling(14, min_periods=mp(14)).mean()
    dn = (-d.clip(upper=0)).rolling(14, min_periods=mp(14)).mean()
    rs_ = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs_)

C["rsi14_slope_20d"] = rsi14(px) - rsi14(px).shift(20)
tr = pd.concat([hi - lo, (hi - px.shift(1)).abs(), (lo - px.shift(1)).abs()], axis=1).max(axis=1)
C["atr_ratio_10x60"] = tr.rolling(10, min_periods=mp(10)).mean() / tr.rolling(60, min_periods=mp(60)).mean().replace(0, np.nan)
C["range_norm_20d"] = ((hi - lo) / px).rolling(20, min_periods=mp(20)).mean()
absmove = ret.abs()
C["price_eff_20d"] = (px / px.shift(20) - 1.0).abs() / absmove.rolling(20, min_periods=mp(20)).sum().replace(0, np.nan)
hi120 = px.rolling(120, min_periods=mp(120)).max()
days_since_hi = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
for c in px.columns:
    is_hi = (px[c] >= hi120[c]).astype(int)
    days_since_hi[c] = is_hi.groupby((is_hi != is_hi.shift()).cumsum()).cumcount()
C["days_since_120d_high_neg"] = -days_since_hi
dvol = vol.pct_change()
C["vol_price_corr_20d"] = ret.rolling(20, min_periods=mp(20)).corr(dvol)
C["mom120_vol20_inter"] = ret120 * (-vol20)
z60 = (ret60 - rm(ret60, 120)) / rs(ret60, 120).replace(0, np.nan)
C["zscore_ret60_120"] = z60

'''
marker = 'print(f"signals built: lib={len(lib)} new={len(C)} ({time.time()-t0:.1f}s)", flush=True)'
assert marker in s, "marker not found"
s = s.replace(marker, insert + '\n' + marker)

open(p, 'w').write(s)
print("patched OK, size", len(s))
