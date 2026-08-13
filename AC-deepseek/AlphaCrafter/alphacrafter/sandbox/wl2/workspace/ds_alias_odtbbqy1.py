import os
# recent candidate results and scripts from 2030
fs = sorted([f for f in os.listdir('scripts') if f.endswith('.py') and ('2030' in f or '2029' in f)])
print('\n'.join(fs[-40:]))
