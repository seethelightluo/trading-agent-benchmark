import json
d = json.load(open('factor_ensemble.json'))
print("selected_factors:", [f["factor_id"] for f in d["selected_factors"]])
print("updated_at:", d["updated_at"])
# check audit tail
import os
print("audit size:", os.path.getsize('factor_library_audit.jsonl'))
