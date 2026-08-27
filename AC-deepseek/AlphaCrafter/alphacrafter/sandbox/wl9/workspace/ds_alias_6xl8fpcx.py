with open('factors/vix_beta_cond_60x20.json') as f:
    d = json.load(f)
    print("Status:", d['validation']['status'])
    print("Last validated:", d['validation']['last_validated'])
    print("IC:", d['validation']['metrics']['ic'])
    print("ICIR:", d['validation']['metrics']['icir'])