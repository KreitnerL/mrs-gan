import matplotlib.pyplot as plt
from ray.tune import Analysis

analysis = Analysis('/home/kreitnerl/mrs-gan/ray_results/pbt_WGP_REG_syn_real_tweak_lambda/')
# Plot by wall-clock time
dfs = analysis.fetch_trial_dataframes()
# This plots everything on the same plot
ax = None
for d in dfs.values():
    ax = d.plot("training_iteration", "score", ax=ax, legend=False)
plt.xlabel("Steps")
plt.ylabel("Relative Absolute Error")
plt.savefig('TEST.png', format='png')