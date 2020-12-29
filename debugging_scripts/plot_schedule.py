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

paths = sorted(make_dataset('/home/kreitnerl/ray_results_25-12-20/pbt_WGP_REG/', ['result.json']))
for i, path in enumerate(paths):
    configs=[]
    with open(path, 'r') as f:
        for line in f:
            step = json.loads(line.rstrip())['config']
            configs.append([step['lambda_A'], step['lambda_feat'], step['dlr'], step['glr']])
    schedule = np.transpose(configs)

    plt.figure()
    for j in range(len(schedule)):
        plt.plot(schedule[j])
    plt.legend(['lambda_A', 'lambda_feat', 'dlr', 'glr'])
    plt.savefig('PBT/PBT_schedule_%d.png'%i, format='png')
    print('saved at PBT/PBT_schedule_%d.png'%i)