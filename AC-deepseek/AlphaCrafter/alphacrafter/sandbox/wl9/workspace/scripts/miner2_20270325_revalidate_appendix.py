# F11: kurt_20d
kurt20 = np.full_like(ret_mat, np.nan)
for i in range(20, T):
    for j in range(N):
        s = ret_mat[i-20:i, j]
        if np.std(s) > 1e-10:
            kurt20[i,j] = pd.Series(s).kurt()
        else:
            kurt20[i,j] = 0

# F12: streak_len_14
streak = np.zeros_like(ret_mat)
for j in range(N):
    c = 0
    for i in range(1, T):
        if ret_mat[i,j] > 0:
            c = c + 1 if ret_mat[i-1,j] > 0 else 1
        elif ret_mat[i,j] < 0:
            c = c - 1 if ret_mat[i-1,j] < 0 else -1
        else:
            c = 0
        streak[i,j] = c

# F13: days_since_high_60 (negative => below recent high)
dsh = np.full_like(close_mat, np.nan)
for i in range(60, T):
    rolling_max = np.max(close_mat[i-60:i], axis=0)
    dsh[i] = close_mat[i] / np.maximum(rolling_max, 1e-10) - 1

# F14: dxy_corr_change_20_60
dxy_corr = np.full_like(ret_mat, np.nan)
if dxy is not None:
    dxy_ret = np.diff(dxy, prepend=0)
    for i in range(60, T):
        for j in range(N):
            r = ret_mat[i-60:i, j]
            d = dxy_ret[i-60:i]
            if np.std(r) > 1e-10 and np.std(d) > 1e-10:
                dxy_corr[i,j] = np.corrcoef(r, d)[0,1]
            else:
                dxy_corr[i,j] = 0

# F15: mom_10_vixreg
mom10_vixreg = np.full_like(ret_mat, np.nan)
if vix is not None:
    vix_hi = np.percentile(vix, 75)
    for i in range(15, T):
        mom10_vixreg[i] = mom10[i]
        if vix[i] > vix_hi:
            mom10_vixreg[i] = -mom10[i]

# F16: vix_beta_cond_60x20
vix_beta_cond = np.full_like(ret_mat, np.nan)
if vix is not None:
    vix_ret2 = np.diff(vix, prepend=0)
    beta20 = np.full_like(ret_mat, np.nan)
    for i in range(20, T):
        rx = ret_mat[i-20:i]
        vx = vix_ret2[i-20:i]
        vx_var = np.var(vx)
        if vx_var > 1e-12:
            cov = np.mean((rx - np.mean(rx, axis=0)) * (vx - np.mean(vx))[:, None], axis=0)
            beta20[i] = cov / vx_var
    for i in range(60, T):
        vix_up = vix_ret2[i] > 0
        vix_beta_cond[i] = beta20[i] if vix_up else beta_vix[i]

# F17: vol_of_vol20x60
vol20 = np.full_like(ret_mat, np.nan)
for i in range(20, T):
    vol20[i] = np.std(ret_mat[i-20:i], axis=0)
vol_of_vol = np.full_like(ret_mat, np.nan)
for i in range(60, T):
    vol_of_vol[i] = np.std(vol20[i-60:i], axis=0)

factor_defs = [
    ("mom_120d_skip5", mom120, 1),
    ("mom_10d_skip5", mom10, 1),
    ("vol_z_20d", vol_z20, 1),
    ("kaufman_eff_20d", kaufman, 1),
    ("bb_width_20d", bbw, 1),
    ("beta_VIX_60", beta_vix, -1),
    ("cny_beta_60", cny_beta, 1),
    ("ac1_120d", ac1, -1),
    ("skew_20d", skew20, 1),
    ("rng_pos_20d", rng_pos, 1),
    ("kurt_20d", kurt20, 1),
    ("streak_len_14", streak, 1),
    ("days_since_high_60", dsh, -1),
    ("dxy_corr_change_20_60", dxy_corr, 1),
    ("mom_10_vixreg", mom10_vixreg, 1),
    ("vix_beta_cond_60x20", vix_beta_cond, -1),
    ("vol_of_vol20x60", vol_of_vol, 1),
]

print(f"\n{'='*120}")
print(f"{'FACTOR':<25} {'IC_mean':>9} {'IC_std':>9} {'ICIR':>9} {'IC_hit%':>8} {'Covg%':>7} {'N_dates':>8} {'Status'}")
print(f"{'='*120}")

results = {}
for fid, fval, direction in factor_defs:
    ic_list = []
    n_valid_dates = 0
    for t in range(1, T):
        if np.isnan(fwd1[t]).any():
            continue
        f_t = fval[t-1]
        r_t = fwd1[t]
        valid = ~(np.isnan(f_t) | np.isnan(r_t))
        nv = np.sum(valid)
        if nv >= 8:
            f_v = f_t[valid]
            r_v = r_t[valid]
            if np.std(f_v) > 1e-10 and np.std(r_v) > 1e-10:
                ic = np.corrcoef(f_v, r_v)[0,1] * direction
                ic_list.append(ic)
                n_valid_dates += 1

    if len(ic_list) > 0:
        ic_mean = np.mean(ic_list)
        ic_std = np.std(ic_list)
        icir = ic_mean / max(ic_std, 1e-10) * np.sqrt(len(ic_list))
        ic_hit = np.mean(np.array(ic_list) > 0) * 100
        covg = np.mean(~np.isnan(fval)) * 100

        passes_ic = abs(ic_mean) >= 0.0070
        passes_icir = abs(icir) >= 0.0840
        if passes_ic and passes_icir:
            status = "EFFECTIVE"
        elif passes_ic:
            status = "WEAK_ICIR"
        else:
            status = "FAIL"

        print(f"{fid:<25} {ic_mean:>9.5f} {ic_std:>9.5f} {icir:>9.4f} {ic_hit:>7.1f}% {covg:>5.1f}% {n_valid_dates:>8} {status}")
        results[fid] = {"ic_mean": round(ic_mean,5), "ic_std": round(ic_std,5), "icir": round(icir,4), 
                        "ic_hit_pct": round(ic_hit,1), "coverage_pct": round(covg,1),
                        "n_valid_dates": n_valid_dates, "status": status}
    else:
        print(f"{fid:<25} {'NO DATA':>9}")

print(f"\n{'='*120}")
print("SUMMARY:")
passing = [k for k,v in results.items() if v['status'] == 'EFFECTIVE']
weak = [k for k,v in results.items() if v['status'] == 'WEAK_ICIR']
fail = [k for k,v in results.items() if v['status'] == 'FAIL']
print(f"  EFFECTIVE: {len(passing)} factors - {passing}")
print(f"  WEAK ICIR: {len(weak)} factors - {weak}")
print(f"  FAIL:      {len(fail)} factors - {fail}")
print(f"  TOTAL:     {len(results)} factors evaluated")