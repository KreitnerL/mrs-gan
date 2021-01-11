import matplotlib.pyplot as plt
import numpy as np
import json
import os

def make_dataset(dir, file_ext=[]):
    paths = []
    assert os.path.exists(dir) and os.path.isdir(dir), '{} is not a valid directory'.format(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            for ext in file_ext:
                if fname.endswith(ext):
                    path = os.path.join(root, fname)
                    paths.append(path)
    return paths

paths = sorted(make_dataset('/home/kreitnerl/mrs-gan/ray_results/pbt_WGP_REG_syn_real_tweak_all', ['result.json']))
configs = []
params = None
for i, path in enumerate(paths):
    configs.append([])
    with open(path, 'r') as f:
        for line in f:
            step: dict = json.loads(line.rstrip())['config']
            configs[-1].append(list(step.values()))
            if params is None:
                params = list(step.keys())
            # configs.append([step['lambda_A'], step['lambda_feat'], step['dlr'], step['glr']])

# configs = (N,L,P)
directory = 'PBT/'
if not os.path.isdir(directory):
    os.mkdir(directory)
max_iter = min(list(map(len, configs)))
configs = np.array(list(map(lambda x: x[:max_iter], configs)))
for i, param in enumerate(params):
    # schedule = np.transpose(configs[:,:,i])
    schedule = np.mean(configs[:,:,i], 0, keepdims=False)
    plt.figure()
    plt.plot(schedule)
    plt.title('Evolution of %s over time' % param)
    plt.xlabel('Steps')
    plt.savefig(os.path.join(directory, 'PBT_shedule_%s.png'%param), format='png', bbox_inches='tight')

print('Done. You can find the schedules at at', directory)