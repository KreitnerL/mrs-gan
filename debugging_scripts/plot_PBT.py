import math
import matplotlib.pyplot as plt
from ray.tune import Analysis

path = '/home/kreitnerl/mrs-gan/ray_results/pbt_WGP_REG_syn_ucsf_tweak_lambda/'
name = path.split('/')[-2]
analysis = Analysis(path)
# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
x = 0
for d in dfs.values():
    x = max(x,max(d.training_iteration))
    ax = d.plot("training_iteration", "score", ax=ax, legend=False)
x = int(math.ceil(x*1.1/10.0))*10
plt.plot(list(range(x)), [0.15]*x, 'r--')
plt.legend([*['_nolegend_']*len(dfs), '15% error mark'])
plt.xlabel("Steps")
plt.ylabel("Relative Absolute Error")
plt.ylim(0, plt.ylim()[1])
# plt.title(name)
plt.savefig('%s.png'%name, format='png', bbox_inches='tight')