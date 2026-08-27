# Continue the script - validation metrics
import json

# Append the rest of the script
with open('scripts/miner_2_20340622_mom_reversal_ratio.py', 'r') as f:
    content = f.read()

# Check if the metric part got cut off
if 'print_mean_ic' not in content:
    additional = """
print(f"\\n=== VALIDATION RESULTS ===")
print(f"Number of valid IC dates: {valid_dates}")
print(f"Mean IC: {mean_ic:.5f}")
print(f"IC std: {std_ic:.5f}")
print(f"ICIR: {icir:.5f}")
print(f"IC hit ratio (pct IC>0): {hit_ratio:.4f}")
print(f"Avg assets per date: {np.mean(n_assets_list):.1f}")
print(f"Min assets per date: {np.min(n_assets_list)}")

# Coverage
coverage = ratio.notna().mean().mean()
print(f"Factor coverage (fraction non-NaN): {coverage:.4f}")

# Turnover - pairwise rank correlation stability
rank_corr_list = []
for i in range(1, len(common_idx)):
    prev_idx = common_idx[i-1]
    cur_idx = common_idx[i]
    prev_row = factor_values.loc[prev_idx]
    cur_row = factor_values.loc[cur_idx]
    valid = prev_row.notna() & cur_row.notna()
    if valid.sum() >= MIN_VALID:
        pv = prev_row[valid].values
        cv = cur_row[valid].values
        if len(np.unique(pv)) > 1 and len(np.unique(cv)) > 1:
            rc, _ = spearmanr(pv, cv)
            if not np.isnan(rc):
                rank_corr_list.append(rc)
if len(rank_corr_list) > 0:
    mean_rank_corr = np.mean(rank_corr_list)
    turnover_10d = 1 - mean_rank_corr
else:
    mean_rank_corr = 0
    turnover_10d = 0
print(f"Mean rank correlation: {mean_rank_corr:.4f}")
print(f"Turnover (1-rankcorr): {turnover_10d:.4f}")

# Admission gate check
IC_THRESH = 0.0070
ICIR_THRESH = 0.0840
passed = (abs(mean_ic) >= IC_THRESH) and (abs(icir) >= ICIR_THRESH)
print(f"\\nAdmission gate: |IC|>={IC_THRESH} and |ICIR|>={ICIR_THRESH}")
print(f"|IC| = {abs(mean_ic):.5f} {'PASS' if abs(mean_ic)>=IC_THRESH else 'FAIL'}")
print(f"|ICIR| = {abs(icir):.5f} {'PASS' if abs(icir)>=ICIR_THRESH else 'FAIL'}")
print(f"OVERALL: {'EFFECTIVE' if passed else 'REJECTED'}")
"""
    with open('scripts/miner_2_20340622_mom_reversal_ratio.py', 'a') as f:
        f.write(additional)
    print("Appended metric calculations")
else:
    print("Already complete")