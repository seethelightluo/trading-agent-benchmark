python - <<'EOF'
import json
sel = ["beta_VIX_60","kaufman_eff_20d","mom_120d_skip5","bb_width_20d","cny_beta_60","vol_z_20d","ac1_120d","mom_10d_skip5","dxy_corr_change_20_60","skew_20d"]
q={}
for f in sel:
    d=json.load(open('factors/'+f+'.json'))
    m=d['validation']['metrics']
    q[f]=abs(m['ic'])*abs(m['icir'])
tot=sum(q.values())
w={f:q[f]/tot for f in sel}
for f in sel:
    print(f, round(w[f],4))
print("sum", round(sum(w.values()),4))
# write ensemble
ens={"schema_version":1,"selected_factors":[{"factor_id":f,"weight":round(w[f],4),"direction":1 if not f in ("beta_VIX_60","ac1_120d") else -1} for f in sel],"method":"quality_ic_tilt"}
json.dump(ens,open('factor_ensemble.json','w'),indent=1)
print("written")
EOF