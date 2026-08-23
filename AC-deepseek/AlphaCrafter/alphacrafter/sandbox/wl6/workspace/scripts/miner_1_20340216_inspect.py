import json, os, glob

# Ensemble
ep = "factor_ensemble.json"
print("=== factor_ensemble.json ===")
if os.path.exists(ep):
    try:
        eg = json.load(open(ep))
        print("selected_factors:", eg.get("selected_factors"))
        print("keys:", list(eg.keys()))
        print("version:", eg.get("version"), "updated:", eg.get("updated"))
    except Exception as e:
        print("ensemble load err:", e)
else:
    print("no ensemble file")

print()
print("=== factor library summaries ===")
files = sorted(glob.glob("factors/*.json"))
for f in files:
    try:
        d = json.load(open(f))
        fid = d.get("factor_id")
        status = (d.get("validation") or {}).get("status")
        lv = d.get("last_validated")
        m = (d.get("validation") or {}).get("metrics") or {}
        ic = m.get("ic"); icir = m.get("icir")
        print(f"{os.path.basename(f):28s} id={str(fid)[:24]:24s} st={status} lv={lv} ic={ic} icir={icir}")
    except Exception as e:
        print(f, "ERR", e)

print()
print("n_files:", len(files))