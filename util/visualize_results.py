from util.util import mkdir
import scipy.io as io
import matplotlib.pyplot as plt
import numpy as np
import os
import util.html as html

def generate_images_of_spectra(items, num, save_dir: str, x):
    plt.figure()
    mkdir(os.path.join(save_dir, 'images'))
    webpage = html.HTML(os.path.join(save_dir), 'Experiment name = %s' % save_dir.split('/')[-1], refresh=1)
    for i in range(num):
        webpage.add_header('Sample [%d]' % i)
        ims, txts, links = [], [], []
        for key,val in items.items():
            if str(key).endswith('spectra'):
                spectrum = np.transpose(val[i])
                plt.plot(x, spectrum)
                plt.title(key[:-8])
                plt.xlabel('ppm')
                plt.xlim([x[0], x[-1]])
                p = os.path.join(save_dir, 'images','{0}_{1}.png'.format(i, key))
                plt.savefig(p, format='png', bbox_inches='tight')
                plt.cla()
                ims.append('{0}_{1}.png'.format(i, key))
                txts.append(key)
                links.append('{0}_{1}.png'.format(i, key))
        webpage.add_images(ims, txts, links, width=256)
    webpage.save()
    print('Done. You can find the generated images in', save_dir)

if __name__ == "__main__":
    path = '/home/kreitnerl/mrs-gan/results/THISISTHETEST/items.mat'
    save_dir = '/home/kreitnerl/mrs-gan/results/THISISTHETEST/'
    items = io.loadmat(path)
    generate_images_of_spectra(items, 2)