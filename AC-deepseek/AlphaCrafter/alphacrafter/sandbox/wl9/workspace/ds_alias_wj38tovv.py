with open('factors/beta_VIX_60.json') as f:
    d = json.load(f)
    print("Status:", d['validation']['status'])
    print("Last validated:", d['validation']['last_validated'])
    print("IC:", d['validation']['metrics']['ic'])
    print("ICIR:", d['validation']['metrics']['icir'])
    print("Expected direction:", d['expected_direction'])