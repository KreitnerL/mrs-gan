import math
import matplotlib.pyplot as plt
from ray.tune import Analysis

path = '/home/kreitnerl/mrs-gan/ray_results/REG-CycleGAN_ucsf_narrow/'
name = path.split('/')[-2]
analysis = Analysis(path)
# analysis.get_best_checkpoint(metric="score", mode="min")
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
plt.ylabel("Mean Relative Error")
plt.ylim(0, plt.ylim()[1])
plt.title('REG-CycleGAN_ucsf PBT validation error over time')
# plt.title(name)
plt.savefig('%s.png'%name, format='png', bbox_inches='tight')