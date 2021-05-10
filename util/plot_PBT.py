import math
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

def plotPBT(path, save_dir: str = None):
    name = path.split('/')[-1]
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'results', name)
    paths = sorted(make_dataset(path, ['result.json']))
    scores = []
    for i, path in enumerate(paths):
        scores.append([])
        with open(path, 'r') as f:
            for line in f:
                step_line = json.loads(line.rstrip())
                scores[-1].append(step_line['score'])
    max_iter = max(list(map(len, scores)))
    plt.figure()
    for i in range(len(scores)):
        plt.plot(scores[i])


    x = int(math.ceil(max_iter*1.1/10.0))*10
    plt.plot(list(range(x)), [0.15]*x, 'r--')
    plt.legend([*['_nolegend_']*len(scores), '15% error mark'])
    plt.xlabel("Steps")
    plt.ylabel("Median Relative Error")
    plt.ylim(bottom=0)
    plt.savefig(os.path.join(save_dir, '%s.png'%name), format='png', bbox_inches='tight')


def plot_pbt_schedule(path: str, save_dir: str = None, file_names: list = ['result.json']):
    if save_dir is None:
        root_dir = os.path.join(os.getcwd(), 'results', path.split('/')[-2])
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        save_dir = os.path.join(root_dir, 'schedule')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    paths = sorted(make_dataset(path, ['result.json']))
    configs = []
    params = None
    for i, path in enumerate(paths):
        configs.append([])
        with open(path, 'r') as f:
            for line in f:
                step_line = json.loads(line.rstrip())
                step: dict = step_line['config']
                configs[-1].append(list(step.values()))
                if params is None:
                    params = list(step.keys())
    max_iter = min(list(map(len, configs)))
    configs = np.array(list(map(lambda x: x[:max_iter], configs)))
    for i, param in enumerate(params):
        mean_schedule = np.mean(configs[:,:,i], 0, keepdims=False)
        # y_std = configs[:,:,i].std(0)
        x = list(range(len(mean_schedule)))
        plt.figure()
        for j in range(len(configs)):
            plt.fill_between(x, configs[j,:,i], mean_schedule, color='#1f77b4', alpha=0.1)
        # plt.fill_between(x, schedule-y_std, schedule+y_std, alpha=0.2)
        plt.plot(mean_schedule, color='#ff7f0e')
        plt.title('Evolution of %s over time' % param)
        plt.xlabel('Steps')
        plt.legend(['Mean', 'Deviation from Mean'])
        plt.savefig(os.path.join(save_dir, 'PBT_schedule_%s.png'%param), format='png', bbox_inches='tight')

    print('Done. You can find the schedules at at', save_dir)

if __name__ == "__main__":
    # plotPBT('/home/kreitnerl/mrs-gan/ray_results/test_feat/')
    plot_pbt_schedule('/home/kreitnerl/mrs-gan/ray_results/REG_CycleGANv2_5/')
