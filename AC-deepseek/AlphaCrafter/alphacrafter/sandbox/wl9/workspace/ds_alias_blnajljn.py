import json
with open('factors/vix_beta_cond_60x20.json') as f:
    d = json.load(f)
    print("Status:", d['validation']['status'])
    print("Last validated:", d['validation']['last_validated'])
    print("IC:", d['validation']['metrics']['ic'])
    print("ICIR:", d['validation']['metrics']['icir'])

with open('factors/beta_VIX_60.json') as f:
    d2 = json.load(f)
    print("\nbeta_VIX_60:")
    print("Status:", d2['validation']['status'])
    print("Last validated:", d2['validation']['last_validated'])
    print("IC:", d2['validation']['metrics']['ic'])
    print("ICIR:", d2['validation']['metrics']['icir'])