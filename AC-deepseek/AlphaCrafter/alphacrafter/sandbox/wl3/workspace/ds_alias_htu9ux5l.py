import json
d = json.load(open('factors/cn10y_beta_60.json'))
print("last_validated:", d['validation']['last_validated'])
print("grid end:", d['signal_artifact_grid']['end'], "n_dates:", d['signal_artifact_grid']['n_dates'])
# check for any runtime/date files
import glob
print(glob.glob('*.json'))
print(glob.glob('*.txt'))
